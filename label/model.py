import os
import json
import boto3
import logging

from uuid import uuid4

from surya.detection import batch_text_detection
from surya.input.pdflines import get_page_text_lines, get_table_blocks
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model, load_processor
from surya.model.layout.model import load_model as load_layout_model
from surya.model.layout.processor import load_processor as load_layout_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.model.table_rec.model import load_model as load_table_model
from surya.model.table_rec.processor import load_processor as load_table_processor
from surya.model.ocr_error.model import load_model as load_ocr_error_model, load_tokenizer as load_ocr_error_processor
from surya.postprocessing.heatmap import draw_polys_on_image, draw_bboxes_on_image
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
from PIL import Image
from surya.languages import CODE_TO_LANGUAGE
from surya.input.langs import replace_lang_with_code
from surya.schema import OCRResult, TextDetectionResult, LayoutResult, TableResult
from surya.settings import settings
from surya.tables import batch_table_recognition
from surya.postprocessing.util import rescale_bbox
from pdftext.extraction import plain_text_output
from surya.ocr_error import batch_ocr_error_detection

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import get_image_size, DATA_UNDEFINED_NAME
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from botocore.exceptions import ClientError
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# load models
det_model, det_processor = load_model(), load_processor()
rec_model, rec_processor = load_rec_model(), load_rec_processor()

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
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

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
        from_name_labels, _, _ = self.get_first_tag_occurence('Polygon', 'Image')
        from_name_trans, _, _ = self.get_first_tag_occurence('TextArea', 'Image')
        labels = self.label_interface.labels
        labels = sum([list(l) for l in labels], [])
        if len(labels) > 1:
            logger.warning('More than one label in the tag. Only the first one will be used: %s', labels[0])
        label = labels[0]

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
        langs = self.LANG_LIST.copy()
        replace_lang_with_code(langs)
        model_results = run_ocr([img_pil], [langs], 
            det_model, det_processor, 
            rec_model, rec_processor, 
            highres_images=[img_pil])[0]

        if not model_results:
            return
        img_width, img_height = get_image_size(image_path)
        result = []
        all_scores = []
        for line in model_results.text_lines:
            if not line:
                logger.warning('Empty result from the model')
                continue
            score = line.confidence
            if score < self.SCORE_THRESHOLD:
                logger.info(f'Skipping result with low score: {score}')
                continue

            rel_pnt = []
            for rp in line.polygon:
                if rp[0] > img_width or rp[1] > img_height:
                    continue
                rel_pnt.append([(rp[0] / img_width) * 100, (rp[1] / img_height) * 100])

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
            # and one for the transcription
            result.append({
                'original_width': img_width,
                'original_height': img_height,
                'image_rotation': 0,
                'value': {
                    'points': rel_pnt,
                    'labels': [label],
                    "text": [line.text]
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
