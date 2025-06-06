import os
import sys

if "APP_PATH" in os.environ:
    app_path = os.path.abspath(os.environ["APP_PATH"])
    if os.getcwd() != app_path:
        # fix sys.path for import
        os.chdir(app_path)
    if app_path not in sys.path:
        sys.path.append(app_path)

import io
import tempfile
from typing import List

import pypdfium2
import gradio as gr

from surya.models import load_predictors

from surya.debug.draw import draw_polys_on_image, draw_bboxes_on_image

from surya.debug.text import draw_text_on_image

from PIL import Image
from surya.recognition.languages import CODE_TO_LANGUAGE, replace_lang_with_code
from surya.table_rec import TableResult
from surya.detection import TextDetectionResult
from surya.recognition import OCRResult
from surya.layout import LayoutResult
from surya.settings import settings
from surya.common.util import rescale_bbox, expand_bbox


# just copy from streamlit_app.py
def run_ocr_errors(pdf_file, page_count, sample_len=512, max_samples=10, max_pages=15):
    from pdftext.extraction import plain_text_output
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(pdf_file.getvalue())
        f.seek(0)

        # Sample the text from the middle of the PDF
        page_middle = page_count // 2
        page_range = range(max(page_middle - max_pages, 0), min(page_middle + max_pages, page_count))
        text = plain_text_output(f.name, page_range=page_range)

    sample_gap = len(text) // max_samples
    if len(text) == 0 or sample_gap == 0:
        return "This PDF has no text or very little text", ["no text"]

    if sample_gap < sample_len:
        sample_gap = sample_len

    # Split the text into samples for the model
    samples = []
    for i in range(0, len(text), sample_gap):
        samples.append(text[i:i + sample_len])

    results = predictors["ocr_error"](samples)
    label = "This PDF has good text."
    if results.labels.count("bad") / len(results.labels) > .2:
        label = "This PDF may have garbled or bad OCR text."
    return label, results.labels

# just copy from streamlit_app.py
def text_detection(img) -> (Image.Image, TextDetectionResult):
    pred = predictors["detection"]([img])[0]
    polygons = [p.polygon for p in pred.bboxes]
    det_img = draw_polys_on_image(polygons, img.copy())
    return det_img, pred

# just copy from streamlit_app.py
def layout_detection(img) -> (Image.Image, LayoutResult):
    pred = predictors["layout"]([img])[0]
    polygons = [p.polygon for p in pred.bboxes]
    labels = [f"{p.label}-{p.position}" for p in pred.bboxes]
    layout_img = draw_polys_on_image(polygons, img.copy(), labels=labels, label_font_size=18)
    return layout_img, pred

# just copy from streamlit_app.py
def table_recognition(img, highres_img, skip_table_detection: bool) -> (Image.Image, List[TableResult]):
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
            # Slightly expand the box
            highres_bbox = expand_bbox(highres_bbox)
            table_imgs.append(
                highres_img.crop(highres_bbox)
            )
            layout_tables.append(highres_bbox)

    table_preds = predictors["table_rec"](table_imgs)
    table_img = img.copy()

    for results, table_bbox in zip(table_preds, layout_tables):
        adjusted_bboxes = []
        labels = []
        colors = []

        for item in results.cells:
            adjusted_bboxes.append([
                (item.bbox[0] + table_bbox[0]),
                (item.bbox[1] + table_bbox[1]),
                (item.bbox[2] + table_bbox[0]),
                (item.bbox[3] + table_bbox[1])
            ])
            labels.append(item.label)
            if "Row" in item.label:
                colors.append("blue")
            else:
                colors.append("red")
        table_img = draw_bboxes_on_image(adjusted_bboxes, highres_img, labels=labels, label_font_size=18, color=colors)
    return table_img, table_preds

# just copy from streamlit_app.py
def ocr(img, highres_img, langs: List[str]) -> (Image.Image, OCRResult):
    replace_lang_with_code(langs)
    img_pred = predictors["recognition"]([img], [langs], predictors["detection"], highres_images=[highres_img])[0]

    bboxes = [l.bbox for l in img_pred.text_lines]
    text = [l.text for l in img_pred.text_lines]
    rec_img = draw_text_on_image(bboxes, text, img.size, langs)
    return rec_img, img_pred

def open_pdf(pdf_file):
    return pypdfium2.PdfDocument(pdf_file)

def page_counter(pdf_file):
    doc = open_pdf(pdf_file)
    doc_len = len(doc)
    doc.close()
    return doc_len

def get_page_image(pdf_file, page_num, dpi=settings.IMAGE_DPI):
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    doc.close()
    return png_image

def get_uploaded_image(in_file):
    return Image.open(in_file).convert("RGB")

# Load models if not already loaded in reload mode
predictors = load_predictors()

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
                count = page_counter(file)
                img = get_page_image(file, num, settings.IMAGE_DPI)
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
            table_img, pred = table_recognition(pil_image, pil_image_highres, skip_table_detection)
            return table_img, [p.model_dump() for p in pred]
        table_rec_btn.click(
            fn=table_rec_img,
            inputs=[in_img, in_file, in_num, use_pdf_boxes_ckb, skip_table_detection_ckb],
            outputs=[result_img, result_json]
        )
        # Run bad PDF text detection
        def ocr_errors_pdf(file, page_count, sample_len=512, max_samples=10, max_pages=15):
            if file.endswith('.pdf'):
                count = page_counter(file)
            else:
                raise gr.Error("This feature only works with PDFs.", duration=5)
            label, results = run_ocr_errors(io.BytesIO(open(file.name, "rb").read()), count)
            return gr.update(label="Result json:" + label, value=results)
        ocr_errors_btn.click(
            fn=ocr_errors_pdf,
            inputs=[in_file, in_num, use_pdf_boxes_ckb, skip_table_detection_ckb],
            outputs=[result_json]
        )

if __name__ == "__main__":
    demo.launch()
