import os
import json
import boto3
import logging

from uuid import uuid4

from surya.inference import SuryaInferenceManager
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor

from PIL import Image

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import get_image_size, DATA_UNDEFINED_NAME
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from botocore.exceptions import ClientError
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# load models
manager = SuryaInferenceManager()
rec_predictor = RecognitionPredictor(manager)
det_predictor = DetectionPredictor()

class SuryaOCR(LabelStudioMLBase):
    """Custom ML Backend model
    """
    LANG_LIST = [lang for lang in os.getenv('LANG_LIST', '').split(',') if lang]

    # score threshold to wipe out noisy results
    SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD', 0.3))
    # file with mappings from COCO labels to custom labels {"airplane": "Boeing"}
    LABEL_MAPPINGS_FILE = os.getenv('LABEL_MAPPINGS_FILE', 'label_mappings.json')

    # Label Studio image upload folder:
    # should be used only in case you use direct file upload into Label Studio instead of URLs
    LABEL_STUDIO_ACCESS_TOKEN = (
        os.environ.get("LABEL_STUDIO_ACCESS_TOKEN") or os.environ.get("LABEL_STUDIO_API_KEY")
    )
    LABEL_STUDIO_HOST = (
        os.environ.get("LABEL_STUDIO_HOST") or os.environ.get("LABEL_STUDIO_URL")
    )

    MODEL_DIR = os.getenv('MODEL_DIR', '.')

    _label_map = {}

    def setup(self):
        """Configure any paramaters of your model here
        """
        self.set("model_version", f'{self.__class__.__name__}-v0.20.0')

        if self.LABEL_MAPPINGS_FILE and os.path.exists(self.LABEL_MAPPINGS_FILE):
            with open(self.LABEL_MAPPINGS_FILE, 'r') as f:
                self._label_map = json.load(f)

    def _get_image_url(self, task, value):
        # TODO: warning! currently only s3 presigned urls are supported with the default keys
        # also it seems not be compatible with file directly uploaded to Label Studio
        # check RND-2 for more details and fix it later
        image_url = task['data'].get(value) or task['data'].get(DATA_UNDEFINED_NAME)

        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url

    def predict_single(self, task):
        logger.debug('Task data: %s', task['data'])
        from_name_poly, to_name, value = self.get_first_tag_occurence('Polygon', 'Image')
        from_name_labels, _, _ = self.get_first_tag_occurence('Labels', 'Image')
        from_name_trans, _, _ = self.get_first_tag_occurence('TextArea', 'Image')

        interface_labels = set(sum([list(l) for l in self.label_interface.labels], []))
        mapped_interface_labels = set(self._label_map.values())
        extra_interface_labels = sorted(interface_labels - mapped_interface_labels)
        if extra_interface_labels:
            logger.warning(
                'Labels configured in Label Studio but missing from label_mappings.json values: %s',
                extra_interface_labels,
            )
        if not mapped_interface_labels:
            logger.warning('No labels configured in label_mappings.json')
        if not self.label_interface.labels:
            logger.warning('No labels configured in interface.')

        image_url = self._get_image_url(task, value)
        cache_dir = os.path.join(self.MODEL_DIR, '.file-cache')
        os.makedirs(cache_dir, exist_ok=True)
        logger.debug(f'Using cache dir: {cache_dir}')
        image_path = get_local_path(
            image_url,
            cache_dir=cache_dir,
            hostname=self.LABEL_STUDIO_HOST,
            access_token=self.LABEL_STUDIO_ACCESS_TOKEN,
            task_id=task.get('id')
        )

        # run ocr
        img_pil = Image.open(image_path).convert("RGB")
        predictions_by_image = rec_predictor([img_pil])
        model_results = predictions_by_image[0]
        if not model_results:
            return

        img_width, img_height = get_image_size(image_path)
        result = []
        all_scores = []
        for block in model_results.blocks:
            if not block:
                logger.warning('Empty result from the model')
                continue
            interface_label = self._label_map.get(block.label)
            if not interface_label:
                logger.info('Skipping result with unmapped label: %s', block.label)
                continue
            if interface_label not in interface_labels:
                logger.info(
                    'Skipping result with label not configured in Label Studio: %s',
                    interface_label,
                )
                continue
            score = block.confidence
            if score < self.SCORE_THRESHOLD:
                logger.info(f'Skipping result with low score: {score}')
                continue

            rel_pnt = []
            for rp in block.polygon:
                if rp[0] > img_width or rp[1] > img_height:
                    continue
                rel_pnt.append([(rp[0] / img_width) * 100, (rp[1] / img_height) * 100])

            text = block.html

            # must add one for the polygon
            id_gen = str(uuid4())[:4]
            result.append({
                'original_width': img_width,
                'original_height': img_height,
                'image_rotation': 0,
                'value': {
                    'points': rel_pnt,
                },
                'id': id_gen,
                'from_name': from_name_poly,
                'to_name': to_name,
                'type': 'polygon',
                'origin': 'manual',
                'score': score,
            })
            # add the region label separately so Label Studio renders it as a label
            result.append({
                'original_width': img_width,
                'original_height': img_height,
                'image_rotation': 0,
                'value': {
                    'labels': [interface_label],
                },
                'id': id_gen,
                'from_name': from_name_labels,
                'to_name': to_name,
                'type': 'labels',
                'origin': 'manual',
                'score': score,
            })
            # and one for the transcription
            result.append({
                'original_width': img_width,
                'original_height': img_height,
                'image_rotation': 0,
                'value': {
                    "text": [text]
                },
                'id': id_gen,
                'from_name': from_name_trans,
                'to_name': to_name,
                'type': 'textarea',
                'origin': 'manual',
                'score': score,
            })
            all_scores.append(score)

        return {
            'result': result,
            'score': sum(all_scores) / max(len(all_scores), 1),
            'model_version': self.get('model_version'),
        }

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        predictions = []
        for task in tasks:
            # TODO: implement is_skipped() function
            # if is_skipped(task):
            #     continue

            prediction = self.predict_single(task)
            if prediction:
                predictions.append(prediction)

        return ModelResponse(predictions=predictions, model_versions=self.get('model_version'))
