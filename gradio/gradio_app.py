import os
import sys

if "APP_PATH" in os.environ:
    app_path = os.path.abspath(os.environ["APP_PATH"])
    if os.getcwd() != app_path:
        # fix sys.path for import
        os.chdir(app_path)
    if app_path not in sys.path:
        sys.path.append(app_path)

import gradio as gr

from typing import List

import pypdfium2
from pypdfium2 import PdfiumError

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


def load_det_cached():
    return load_model(), load_processor()

def load_rec_cached():
    return load_rec_model(), load_rec_processor()

def load_layout_cached():
    return load_layout_model(), load_layout_processor()

def load_table_cached():
    return load_table_model(), load_table_processor()

def load_ocr_error_cached():
    return load_ocr_error_model(), load_ocr_error_processor()

#
def run_ocr_errors(pdf_file, page_count, sample_len=512, max_samples=10, max_pages=15):
    # Sample the text from the middle of the PDF
    page_middle = page_count // 2
    page_range = range(max(page_middle - max_pages, 0), min(page_middle + max_pages, page_count))
    text = plain_text_output(pdf_file, page_range=page_range)

    sample_gap = len(text) // max_samples
    if len(text) == 0 or sample_gap == 0:
        return "This PDF has no text or very little text", ["no text"]

    if sample_gap < sample_len:
        sample_gap = sample_len

    # Split the text into samples for the model
    samples = []
    for i in range(0, len(text), sample_gap):
        samples.append(text[i:i + sample_len])

    results = batch_ocr_error_detection(samples, ocr_error_model, ocr_error_processor)
    label = "This PDF has good text."
    if results.labels.count("bad") / len(results.labels) > .2:
        label = "This PDF may have garbled or bad OCR text."
    return label, results.labels

#
def text_detection(img) -> (Image.Image, TextDetectionResult):
    pred = batch_text_detection([img], det_model, det_processor)[0]
    polygons = [p.polygon for p in pred.bboxes]
    det_img = draw_polys_on_image(polygons, img.copy())
    return det_img, pred

#
def layout_detection(img) -> (Image.Image, LayoutResult):
    pred = batch_layout_detection([img], layout_model, layout_processor)[0]
    polygons = [p.polygon for p in pred.bboxes]
    labels = [f"{p.label}-{p.position}" for p in pred.bboxes]
    layout_img = draw_polys_on_image(polygons, img.copy(), labels=labels, label_font_size=18)
    return layout_img, pred

#
def table_recognition(img, highres_img, filepath, page_idx: int, use_pdf_boxes: bool, skip_table_detection: bool) -> (Image.Image, List[TableResult]):
    if skip_table_detection:
        layout_tables = [(0, 0, highres_img.size[0], highres_img.size[1])]
        table_imgs = [highres_img]
    else:
        _, layout_pred = layout_detection(img)
        layout_tables_lowres = [l.bbox for l in layout_pred.bboxes if l.label == "Table"]
        table_imgs = []
        layout_tables = []
        for tb in layout_tables_lowres:
            highres_bbox = rescale_bbox(tb, img.size, highres_img.size)
            table_imgs.append(
                highres_img.crop(highres_bbox)
            )
            layout_tables.append(highres_bbox)

    try:
        page_text = get_page_text_lines(filepath, [page_idx], [highres_img.size])[0]
        table_bboxes = get_table_blocks(layout_tables, page_text, highres_img.size)
    except PdfiumError:
        # This happens when we try to get text from an image
        table_bboxes = [[] for _ in layout_tables]

    if not use_pdf_boxes or any(len(tb) == 0 for tb in table_bboxes):
        det_results = batch_text_detection(table_imgs, det_model, det_processor)
        table_bboxes = [[{"bbox": tb.bbox, "text": None} for tb in det_result.bboxes] for det_result in det_results]

    table_preds = batch_table_recognition(table_imgs, table_bboxes, table_model, table_processor)
    table_img = img.copy()

    for results, table_bbox in zip(table_preds, layout_tables):
        adjusted_bboxes = []
        labels = []
        colors = []

        for item in results.rows + results.cols:
            adjusted_bboxes.append([
                (item.bbox[0] + table_bbox[0]),
                (item.bbox[1] + table_bbox[1]),
                (item.bbox[2] + table_bbox[0]),
                (item.bbox[3] + table_bbox[1])
            ])
            labels.append(item.label)
            if hasattr(item, "row_id"):
                colors.append("blue")
            else:
                colors.append("red")
        table_img = draw_bboxes_on_image(adjusted_bboxes, highres_img, labels=labels, label_font_size=18, color=colors)
    return table_img, table_preds

# Function for OCR
def ocr(img, highres_img, langs: List[str]) -> (Image.Image, OCRResult):
    replace_lang_with_code(langs)
    img_pred = run_ocr([img], [langs], det_model, det_processor, rec_model, rec_processor, highres_images=[highres_img])[0]

    bboxes = [l.bbox for l in img_pred.text_lines]
    text = [l.text for l in img_pred.text_lines]
    rec_img = draw_text_on_image(bboxes, text, img.size, langs, has_math="_math" in langs)
    return rec_img, img_pred

def open_pdf(pdf_file):
    return pypdfium2.PdfDocument(pdf_file)

def count_pdf(pdf_file):
    doc = open_pdf(pdf_file)
    return len(doc)

def get_page_image(pdf_file, page_num, dpi=96):
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    return png_image

def get_uploaded_image(in_file):
    return Image.open(in_file).convert("RGB")

# Load models if not already loaded in reload mode
if 'det_model' not in globals():
    det_model, det_processor = load_det_cached()
    rec_model, rec_processor = load_rec_cached()
    layout_model, layout_processor = load_layout_cached()
    table_model, table_processor = load_table_cached()
    ocr_error_model, ocr_error_processor = load_ocr_error_cached()

with gr.Blocks(title="Surya") as demo:
    gr.Markdown("""
    # Surya OCR Demo

    This app will let you try surya, a multilingual OCR model. It supports text detection + layout analysis in any language, and text recognition in 90+ languages.

    Notes:
    - This works best on documents with printed text.
    - Preprocessing the image (e.g. increasing contrast) can improve results.
    - If OCR doesn't work, try changing the resolution of your image (increase if below 2048px width, otherwise decrease).
    - This supports 90+ languages, see [here](https://github.com/VikParuchuri/surya/tree/master/surya/languages.py) for a full list.

    Find the project [here](https://github.com/VikParuchuri/surya).
    """)

    with gr.Row():
        with gr.Column():
            in_file = gr.File(label="PDF file or image:", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp"])
            in_num = gr.Slider(label="Page number", minimum=1, maximum=100, value=1, step=1)
            in_img = gr.Image(label="Select page of Image", type="pil", sources=None)

            text_det_btn = gr.Button("Run Text Detection")
            layout_det_btn = gr.Button("Run Layout Analysis")

            lang_dd = gr.Dropdown(label="Languages", choices=sorted(list(CODE_TO_LANGUAGE.values())), multiselect=True, max_choices=4, info="Select the languages in the image (if known) to improve OCR accuracy.  Optional.")
            text_rec_btn = gr.Button("Run OCR")

            use_pdf_boxes_ckb = gr.Checkbox(label="Use PDF table boxes", value=True, info="Table recognition only: Use the bounding boxes from the PDF file vs text detection model.")
            skip_table_detection_ckb = gr.Checkbox(label="Skip table detection", value=False, info="Table recognition only: Skip table detection and treat the whole image/page as a table.")
            table_rec_btn = gr.Button("Run Table Rec")

            ocr_errors_btn = gr.Button("Run bad PDF text detection")
        with gr.Column():
            result_img = gr.Image(label="Result image")
            result_json = gr.JSON(label="Result json")

        def show_image(file, num=1):
            if file.endswith('.pdf'):
                count = count_pdf(file)
                img = get_page_image(file, num)
                return [
                    gr.update(visible=True, maximum=count),
                    gr.update(value=img)]
            else:
                img = get_uploaded_image(file)
                return [
                    gr.update(visible=False),
                    gr.update(value=img)]

        in_file.upload(
            fn=show_image,
            inputs=[in_file],
            outputs=[in_num, in_img],
        )
        in_num.change(
            fn=show_image,
            inputs=[in_file, in_num],
            outputs=[in_num, in_img],
        )

        # Run Text Detection
        def text_det_img(pil_image):
            det_img, pred = text_detection(pil_image)
            return det_img, pred.model_dump(exclude=["heatmap", "affinity_map"])
        text_det_btn.click(
            fn=text_det_img,
            inputs=[in_img],
            outputs=[result_img, result_json]
        )
        # Run layout
        def layout_det_img(pil_image):
            layout_img, pred = layout_detection(pil_image)
            return layout_img, pred.model_dump(exclude=["segmentation_map"])
        layout_det_btn.click(
            fn=layout_det_img,
            inputs=[in_img],
            outputs=[result_img, result_json]
        )
        # Run OCR
        def text_rec_img(pil_image, in_file, page_number, languages):
            if in_file.endswith('.pdf'):
                pil_image_highres = get_page_image(in_file, page_number, dpi=settings.IMAGE_DPI_HIGHRES)
            else:
                pil_image_highres = pil_image
            rec_img, pred = ocr(pil_image, pil_image_highres, languages)
            return rec_img, pred.model_dump()
        text_rec_btn.click(
            fn=text_rec_img,
            inputs=[in_img, in_file, in_num, lang_dd],
            outputs=[result_img, result_json]
        )
        # Run Table Recognition
        def table_rec_img(pil_image, in_file, page_number, use_pdf_boxes, skip_table_detection):
            if in_file.endswith('.pdf'):
                pil_image_highres = get_page_image(in_file, page_number, dpi=settings.IMAGE_DPI_HIGHRES)
            else:
                pil_image_highres = pil_image
            table_img, pred = table_recognition(pil_image, pil_image_highres, in_file, page_number - 1 if page_number else None, use_pdf_boxes, skip_table_detection)
            return table_img, [p.model_dump() for p in pred]
        table_rec_btn.click(
            fn=table_rec_img,
            inputs=[in_img, in_file, in_num, use_pdf_boxes_ckb, skip_table_detection_ckb],
            outputs=[result_img, result_json]
        )
        # Run bad PDF text detection
        def ocr_errors_pdf(file, page_count, sample_len=512, max_samples=10, max_pages=15):
            if file.endswith('.pdf'):
                count = count_pdf(file)
            else:
                raise gr.Error("This feature only works with PDFs.", duration=5)
            label, results = run_ocr_errors(file, count)
            return gr.update(label="Result json:" + label, value=results)
        ocr_errors_btn.click(
            fn=ocr_errors_pdf,
            inputs=[in_file, in_num, use_pdf_boxes_ckb, skip_table_detection_ckb],
            outputs=[result_json]
        )

if __name__ == "__main__":
    demo.launch()
