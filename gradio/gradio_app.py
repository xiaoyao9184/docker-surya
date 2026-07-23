import llama_server  # noqa: F401

import os
import sys

if "APP_PATH" in os.environ:
    app_path = os.path.abspath(os.environ["APP_PATH"])
    if os.getcwd() != app_path:
        # fix sys.path for import
        os.chdir(app_path)
    if app_path not in sys.path:
        sys.path.append(app_path)

import base64
import io
import html as html_lib
import re
import tempfile
import time
from typing import List

import pypdfium2
import gradio as gr
import requests
from contextlib import suppress
from PIL import Image, ImageDraw

from surya.debug.draw import draw_polys_on_image, draw_bboxes_on_image
from surya.detection import TextDetectionResult
from surya.inference import SuryaInferenceManager
from surya.layout import LayoutPredictor
from surya.layout.schema import LayoutResult
from surya.recognition import RecognitionPredictor
from surya.recognition.schema import PageOCRResult
from surya.settings import settings
from surya.table_rec import TableRecPredictor
from surya.table_rec.schema import TableResult

# just copy from streamlit_app.py
# KaTeX-enabled HTML wrapper. The OCR HTML wraps math in <math>...</math>
# (KaTeX-compatible LaTeX inside), which a browser would otherwise show as
# raw text. We convert those tags to \( \) / \[ \] delimiters and let KaTeX
# auto-render typeset them inside an iframe.
_KATEX_HEAD = r"""<!doctype html><html><head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
html,body{background:#ffffff;}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:15px;line-height:1.55;color:#111111;margin:0;padding:14px;}
table{border-collapse:collapse;margin:6px 0;} td,th{border:1px solid #bbb;padding:3px 6px;color:#111111;}
[data-label="SectionHeader"],[data-label="PageHeader"]{font-weight:600;}
</style></head><body>
"""
# just copy from streamlit_app.py
_KATEX_TAIL = r"""
<script>
renderMathInElement(document.body, {
  delimiters: [
    {left: "\\[", right: "\\]", display: true},
    {left: "\\(", right: "\\)", display: false}
  ],
  throwOnError: false
});
</script></body></html>
"""
# just copy from streamlit_app.py
_MATH_RE = re.compile(r"<math\b([^>]*)>(.*?)</math>", re.DOTALL | re.IGNORECASE)

# just copy from streamlit_app.py
def _math_to_katex(html_str: str) -> str:
    """Rewrite <math>...</math> tags into KaTeX \\( \\) / \\[ \\] delimiters."""

    def repl(m: "re.Match") -> str:
        attrs, inner = m.group(1), m.group(2)
        if re.search(r"""display\s*=\s*["']block["']""", attrs):
            return "\\[" + inner + "\\]"
        return "\\(" + inner + "\\)"

    return _MATH_RE.sub(repl, html_str or "")


def render_ocr_html(html_str: str, height: int = 400) -> str:
    """Render OCR HTML with math typeset by KaTeX in a Gradio HTML iframe."""
    doc = _KATEX_HEAD + _math_to_katex(html_str) + _KATEX_TAIL
    srcdoc = html_lib.escape(doc, quote=True)
    return (
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%;height:{height}px;border:1px solid #ddd;background:#fff;" '
        'sandbox="allow-scripts"></iframe>'
    )


# just copy from streamlit_app.py
def _assemble_page_html(page: PageOCRResult) -> str:
    """Reconstruct a div-block whole-page HTML from a PageOCRResult."""
    parts: List[str] = []
    for blk in page.blocks:
        if blk.skipped:
            continue
        x0, y0, x1, y1 = (int(c) for c in blk.bbox)
        body = blk.html or ""
        parts.append(
            f'<div data-bbox="{x0} {y0} {x1} {y1}" data-label="{blk.label}">{body}</div>'
        )
    return "\n".join(parts)

# just copy from streamlit_app.py
def text_detection(img) -> tuple[Image.Image, TextDetectionResult, float]:
    t = time.perf_counter()
    text_pred = predictors["detection"]([img])[0]
    elapsed = time.perf_counter() - t
    text_polygons = [p.polygon for p in text_pred.bboxes]
    det_img = draw_polys_on_image(text_polygons, img.copy())
    return det_img, text_pred, elapsed

# just copy from streamlit_app.py
def layout_detection(img) -> tuple[Image.Image, LayoutResult, float]:
    t = time.perf_counter()
    pred = predictors["layout"]([img])[0]
    elapsed = time.perf_counter() - t
    polygons = [p.polygon for p in pred.bboxes]
    labels = [
        f"{p.label}-{p.position}-c{p.count}-{round(p.confidence or 0, 2)}"
        for p in pred.bboxes
    ]
    annotated = draw_polys_on_image(
        polygons, img.copy(), labels=labels, label_font_size=14
    )
    return annotated, pred, elapsed

# just copy from streamlit_app.py
def block_ocr(img) -> tuple[Image.Image, PageOCRResult, LayoutResult, float, float]:
    """Layout → block crops → BLOCK_PROMPT. Returns layout + block-OCR timings."""
    t_layout = time.perf_counter()
    layout = predictors["layout"]([img])[0]
    layout_elapsed = time.perf_counter() - t_layout

    t_blocks = time.perf_counter()
    page_results = predictors["recognition"]([img], [layout])
    blocks_elapsed = time.perf_counter() - t_blocks
    page = page_results[0]

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    for blk in page.blocks:
        x0, y0, x1, y1 = blk.bbox
        color = "red" if blk.error else ("orange" if blk.skipped else "green")
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), f"{blk.reading_order} {blk.label}", fill=color)
    return annotated, page, layout, layout_elapsed, blocks_elapsed

# just copy from streamlit_app.py
def full_page_ocr(img) -> tuple[Image.Image, PageOCRResult, float]:
    """Single HIGH_ACCURACY_BBOX_PROMPT call on the whole page."""
    t = time.perf_counter()
    page_results = predictors["recognition"]([img], full_page=True)
    elapsed = time.perf_counter() - t
    page = page_results[0]
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    for blk in page.blocks:
        x0, y0, x1, y1 = blk.bbox
        color = "red" if blk.error else ("orange" if blk.skipped else "green")
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), f"{blk.reading_order} {blk.label}", fill=color)
    return annotated, page, elapsed

# just copy from streamlit_app.py
def table_recognition(
    img: Image.Image,
    mode: str,
    skip_table_detection: bool,
) -> tuple[Image.Image, List[TableResult], float, float]:
    """Returns (annotated_img, table_preds, layout_elapsed, table_rec_elapsed)."""
    layout_elapsed = 0.0
    if skip_table_detection:
        table_imgs = [img]
        table_counts = [0]
        table_bboxes = [(0, 0, img.size[0], img.size[1])]
    else:
        t = time.perf_counter()
        layout = predictors["layout"]([img])[0]
        layout_elapsed = time.perf_counter() - t
        tables = [b for b in layout.bboxes if b.label in ("Table", "TableOfContents")]
        if not tables:
            return img.copy(), [], layout_elapsed, 0.0
        table_bboxes = [tuple(int(c) for c in b.bbox) for b in tables]
        table_imgs = [img.crop(b) for b in table_bboxes]
        table_counts = [b.count for b in tables]

    t = time.perf_counter()
    if mode == "full":
        table_preds = predictors["table_rec"].predict_full(
            table_imgs, counts=table_counts
        )
    else:
        table_preds = predictors["table_rec"].predict_simple(table_imgs)
    table_rec_elapsed = time.perf_counter() - t

    out_img = img.copy()
    for pred, table_img, tbbox in zip(table_preds, table_imgs, table_bboxes):
        if pred.error or pred.mode != "simple" or not pred.rows:
            continue
        row_bboxes = [r.bbox for r in pred.rows]
        col_bboxes = [c.bbox for c in pred.cols]
        row_labels = [r.label for r in pred.rows]
        col_labels = [c.label for c in pred.cols]
        annot = table_img.copy()
        annot = draw_bboxes_on_image(
            row_bboxes, annot, labels=row_labels, label_font_size=14, color="blue"
        )
        annot = draw_bboxes_on_image(
            col_bboxes, annot, labels=col_labels, label_font_size=14, color="red"
        )
        # Paste annotated crop back at the table's position in the page.
        out_img.paste(annot, (tbbox[0], tbbox[1]))
    return out_img, table_preds, layout_elapsed, table_rec_elapsed

# just copy from streamlit_app.py
def ocr_errors(pdf_file, page_count, sample_len=512, max_samples=10, max_pages=15):
    from pdftext.extraction import plain_text_output

    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(pdf_file.getvalue())
        f.seek(0)

        page_middle = page_count // 2
        page_range = range(
            max(page_middle - max_pages, 0), min(page_middle + max_pages, page_count)
        )
        text = plain_text_output(f.name, page_range=page_range)

    sample_gap = len(text) // max_samples
    if len(text) == 0 or sample_gap == 0:
        return "This PDF has no text or very little text", ["no text"]

    if sample_gap < sample_len:
        sample_gap = sample_len

    samples = []
    for i in range(0, len(text), sample_gap):
        samples.append(text[i : i + sample_len])

    results = predictors["ocr_error"](samples)
    label = "This PDF has good text."
    if results.labels.count("bad") / len(results.labels) > 0.2:
        label = "This PDF may have garbled or bad OCR text."
    return label, results.labels

# just copy from streamlit_app.py
def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)

# just copy from streamlit_app.py
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

# just copy from streamlit_app.py
def page_counter(pdf_file):
    doc = open_pdf(pdf_file)
    doc_len = len(doc)
    doc.close()
    return doc_len


class GradioUploadedFile:
    def __init__(self, file):
        self.path = file.name if hasattr(file, "name") else file
        self.name = self.path

    def getvalue(self):
        with open(self.path, "rb") as f:
            return f.read()

    def endswith(self, suffix):
        return self.path.endswith(suffix)


def _image_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _block_details_html(page: PageOCRResult, img: Image.Image | None = None) -> str:
    parts: List[str] = [
        "<div style=\"display:flex;flex-direction:column;gap:12px;\">"
    ]
    for blk in page.blocks:
        confidence = blk.confidence if blk.confidence is not None else 0
        title = html_lib.escape(
            f"#{blk.reading_order} {blk.label} (conf {confidence:.2f})"
        )
        parts.append(
            "<details style=\"border:1px solid #ddd;border-radius:6px;padding:10px;\">"
            f"<summary style=\"cursor:pointer;font-weight:600;\">{title}</summary>"
        )

        if img is not None and blk.polygon:
            xs = [p[0] for p in blk.polygon]
            ys = [p[1] for p in blk.polygon]
            bbox_drawn = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            cx0 = max(0, int(min(xs)) - 4)
            cy0 = max(0, int(min(ys)) - 4)
            cx1 = min(img.size[0], int(max(xs)) + 4)
            cy1 = min(img.size[1], int(max(ys)) + 4)
            parts.append(
                "<pre style=\"white-space:pre-wrap;padding:8px;border-radius:4px;\">"
                f"bbox(drawn) = {bbox_drawn}\n"
                f"crop(ocr)  = {(cx0, cy0, cx1, cy1)}  (= bbox +/- 4px pad)"
                "</pre>"
            )

            thumb = img.copy()
            ImageDraw.Draw(thumb).rectangle(bbox_drawn, outline="red", width=4)
            parts.append(
                "<div style=\"display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;align-items:start;\">"
                "<figure style=\"margin:0;\"><figcaption>this block's drawn rect (red)</figcaption>"
                f"<img src=\"{_image_data_uri(thumb)}\" style=\"width:100%;max-width:300px;height:auto;\" /></figure>"
            )
            if cx1 > cx0 and cy1 > cy0:
                crop = img.crop((cx0, cy0, cx1, cy1))
                parts.append(
                    "<figure style=\"margin:0;\"><figcaption>OCR crop</figcaption>"
                    f"<img src=\"{_image_data_uri(crop)}\" style=\"width:100%;height:auto;\" /></figure>"
                )
            parts.append("</div>")

        if blk.skipped:
            parts.append("<p style=\"color:#9a6700;\">Block skipped (visual label)</p>")
        elif blk.error:
            parts.append("<p style=\"color:#b42318;\">Block OCR errored</p>")
        else:
            blk_html = blk.html or ""
            parts.append(render_ocr_html(blk_html, height=160))
            parts.append(
                "<pre style=\"white-space:pre-wrap;padding:8px;border-radius:4px;overflow:auto;\">"
                f"{html_lib.escape(blk_html)}"
                "</pre>"
            )

        parts.append("</details>")
    parts.append("</div>")
    return "\n".join(parts)


def load_predictors():
    manager = SuryaInferenceManager()
    layout_predictor = LayoutPredictor(manager)
    rec_predictor = RecognitionPredictor(manager)
    table_rec_predictor = TableRecPredictor(manager)

    # Lazy-import detection / ocr_error to keep startup snappy when the user
    # only wants VLM modes
    from surya.detection import DetectionPredictor
    from surya.ocr_error import OCRErrorPredictor

    return {
        "manager": manager,
        "layout": layout_predictor,
        "recognition": rec_predictor,
        "table_rec": table_rec_predictor,
        "detection": DetectionPredictor(),
        "ocr_error": OCRErrorPredictor(),
    }


# Load models if not already loaded in reload mode
predictors = load_predictors()

with gr.Blocks(title="Surya") as demo:
    gr.Markdown("""
    # Surya 2 Demo

    This app will let you try surya, a multilingual OCR model. It supports text detection + layout analysis in any language, and text recognition in 90+ languages.

    Notes:
    - This works best on documents with printed text.
    - Preprocessing the image (e.g. increasing contrast) can improve results.
    - If OCR doesn't work, try changing the resolution of your image (increase if below 2048px width, otherwise decrease).
    - This supports 90+ languages, see [here](https://github.com/VikParuchuri/surya/tree/master/surya/languages.py) for a full list.

    Find the original project [here](https://github.com/VikParuchuri/surya).
    Or this project [here](https://github.com/xiaoyao9184/docker-surya).
    See the [README](./blob/main/README.md) for Spaces's metadata.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            in_file = gr.File(label="PDF file or image:", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp"])
            in_num = gr.Slider(label="Page number", minimum=1, maximum=100, value=1, step=1)
            text_det_btn = gr.Button("Run Text Detection")
            layout_det_btn = gr.Button("Run Layout Analysis")

            block_ocr_btn = gr.Button("Run Block OCR")
            full_page_ocr_btn = gr.Button("Run Full-Page OCR")

            table_mode_rad = gr.Radio(
                label="Table mode",
                choices=["simple", "full"],
                value="simple",
                info="simple: rows+cols only. full: full HTML.",
            )
            skip_table_detection_ckb = gr.Checkbox(label="Skip table detection", value=False, info="Table recognition only: Skip table detection and treat the whole image/page as a table.")
            table_rec_btn = gr.Button("Run Table Rec")

            ocr_errors_btn = gr.Button("Run bad PDF text detection")

            result_html_rendered = gr.HTML(label="Result html")
            result_block_details = gr.HTML(label="Block details")
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    in_img = gr.Image(label="Select page of Image", type="pil", sources=None)
                with gr.Column():
                    result_img = gr.Image(label="Result image")

            with gr.Row():
                result_json = gr.JSON(label="Result json", max_height=None)
            with gr.Row():
                result_html_source = gr.Code(label="Result html", language='html')

        def show_image(file, num=1):
            if file.endswith('.pdf'):
                pdf_file = GradioUploadedFile(file)
                count = page_counter(pdf_file)
                img = get_page_image(pdf_file, num, settings.IMAGE_DPI)
                return [
                    gr.update(visible=True, maximum=count),
                    gr.update(value=img)]
            else:
                img = Image.open(file).convert("RGB")
                return [
                    gr.update(visible=False),
                    gr.update(value=img)]

        def get_image(file, num):
            if file.endswith('.pdf'):
                pdf_file = GradioUploadedFile(file)
                return get_page_image(pdf_file, num, dpi=settings.IMAGE_DPI_HIGHRES)
            else:
                return Image.open(file).convert("RGB")

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
        def text_det_img(in_file, page_number):
            # update counter
            with suppress(Exception):
                requests.get("https://counterapi.com/api/xiaoyao9184.github.com/view/docker-surya")

            pil_image_highres = get_image(in_file, page_number)
            det_img, pred, elapsed = text_detection(pil_image_highres)
            det_json = pred.model_dump(exclude=["heatmap", "affinity_map"])
            return (
                gr.update(label="Result image: text detected", value=det_img),
                gr.update(label=f"Result json: {elapsed * 1000:.0f} ms ({elapsed:.2f}s) {len(pred.bboxes)} polys text boxes detected", value=det_json)
            )
        text_det_btn.click(
            fn=text_det_img,
            inputs=[in_file, in_num],
            outputs=[result_img, result_json]
        )

        # Run layout
        def layout_det_img(in_file, page_number):
            # update counter
            with suppress(Exception):
                requests.get("https://counterapi.com/api/xiaoyao9184.github.com/view/docker-surya")

            pil_image_highres = get_image(in_file, page_number)
            layout_img, pred, elapsed = layout_detection(pil_image_highres)
            layout_json = pred.model_dump(exclude=["segmentation_map"])
            return (
                gr.update(label="Result image: layout detected", value=layout_img),
                gr.update(label=f"Result json: {elapsed * 1000:.0f} ms ({elapsed:.2f}s) {len(pred.bboxes)} polys layout labels detected", value=layout_json)
            )
        layout_det_btn.click(
            fn=layout_det_img,
            inputs=[in_file, in_num],
            outputs=[result_img, result_json]
        )

        # Run Block OCR
        def block_ocr_img(in_file, page_number):
            # update counter
            with suppress(Exception):
                requests.get("https://counterapi.com/api/xiaoyao9184.github.com/view/docker-surya")

            pil_image_highres = get_image(in_file, page_number)
            annotated, page, layout, t_layout, t_blocks = block_ocr(pil_image_highres)
            n_blocks = len(page.blocks)
            n_ok = sum(1 for b in page.blocks if not b.skipped and not b.error)
            total = t_layout + t_blocks
            timing = (
                f"Block OCR: layout {t_layout * 1000:.0f} ms ({t_layout:.2f}s), "
                f"{n_blocks} blocks; per-block OCR {t_blocks * 1000:.0f} ms "
                f"({t_blocks:.2f}s), {n_ok} OCR'd; total {total * 1000:.0f} ms "
                f"({total:.2f}s)"
            )
            full_html = _assemble_page_html(page)
            return (
                gr.update(label="Result image: " + "Block OCR (green=ok, orange=skipped, red=error)", value=annotated),
                gr.update(label="Result json: " + timing, value=page.model_dump()),
                gr.update(label="Result HTML source: full page HTML", value=full_html),
                gr.update(label="Result HTML rendered: full page HTML", value=render_ocr_html(full_html, height=600)),
                gr.update(label="Block details", value=_block_details_html(page, pil_image_highres)),
            )
        block_ocr_btn.click(
            fn=block_ocr_img,
            inputs=[in_file, in_num],
            outputs=[result_img, result_json, result_html_source, result_html_rendered, result_block_details]
        )

        # Run Full OCR
        def full_page_ocr_img(in_file, page_number):
            # update counter
            with suppress(Exception):
                requests.get("https://counterapi.com/api/xiaoyao9184.github.com/view/docker-surya")

            pil_image_highres = get_image(in_file, page_number)
            annotated, page, elapsed = full_page_ocr(pil_image_highres)
            n_blocks = len(page.blocks)
            n_ok = sum(1 for b in page.blocks if not b.skipped and not b.error)
            timing = (
                f"Full-Page OCR: {elapsed * 1000:.0f} ms ({elapsed:.2f}s), "
                f"{n_blocks} blocks parsed, {n_ok} OK"
            )
            full_html = _assemble_page_html(page)
            return (
                gr.update(label="Result image: " + "Full-Page (green=ok, orange=skipped, red=error)", value=annotated),
                gr.update(label="Result json: " + timing, value=page.model_dump()),
                gr.update(label="Result HTML source: full page HTML", value=full_html),
                gr.update(label="Result HTML rendered: full page HTML", value=render_ocr_html(full_html, height=600)),
                gr.update(label="Block details", value=_block_details_html(page, pil_image_highres)),
            )
        full_page_ocr_btn.click(
            fn=full_page_ocr_img,
            inputs=[in_file, in_num],
            outputs=[result_img, result_json, result_html_source, result_html_rendered, result_block_details]
        )

        # Run Table Recognition
        def table_rec_img(in_file, page_number, table_mode, skip_table_detection):
            # update counter
            with suppress(Exception):
                requests.get("https://counterapi.com/api/xiaoyao9184.github.com/view/docker-surya")

            pil_image_highres = get_image(in_file, page_number)
            table_img, pred, t_layout, t_table = table_recognition(pil_image_highres, table_mode, skip_table_detection)
            table_json = [p.model_dump() for p in pred]
            timing_parts = []
            if not skip_table_detection:
                timing_parts.append(
                    f"layout {t_layout * 1000:.0f} ms ({t_layout:.2f}s), {len(pred)} tables found"
                )
            timing_parts.append(f"{table_mode} {t_table * 1000:.0f} ms ({t_table:.2f}s)")
            if not skip_table_detection:
                total = t_layout + t_table
                timing_parts.append(f"total {total * 1000:.0f} ms ({total:.2f}s)")

            table_html = "\n\n".join(
                p.html for p in pred if p.mode == "full" and p.html
            )

            return (
                gr.update(label="Result image: table recognized", value=table_img),
                gr.update(
                    label=f"Result json: {'; '.join(timing_parts)}",
                    value=table_json,
                ),
                gr.update(
                    label="Result HTML source: table HTML" if table_html else "Result HTML source: NONE",
                    value=table_html,
                ),
                gr.update(
                    label="Result HTML rendered: table HTML" if table_html else "Result HTML rendered: NONE",
                    value=render_ocr_html(table_html, height=400) if table_html else "",
                ),
                gr.update(label="Block details", value=""),
            )
        table_rec_btn.click(
            fn=table_rec_img,
            inputs=[in_file, in_num, table_mode_rad, skip_table_detection_ckb],
            outputs=[result_img, result_json, result_html_source, result_html_rendered, result_block_details]
        )

        # Run bad PDF text detection
        def ocr_errors_pdf(in_file):
            # update counter
            with suppress(Exception):
                requests.get("https://counterapi.com/api/xiaoyao9184.github.com/view/docker-surya")

            if not in_file.endswith('.pdf'):
                raise gr.Error("This feature only works with PDFs.", duration=5)
            pdf_file = GradioUploadedFile(in_file)
            page_count = page_counter(pdf_file)
            layout_label, layout_json = ocr_errors(pdf_file, page_count)
            return (
                gr.update(label="Result image: NONE", value=None),
                gr.update(label="Result json: " + layout_label, value=layout_json)
            )
        ocr_errors_btn.click(
            fn=ocr_errors_pdf,
            inputs=[in_file],
            outputs=[result_img, result_json]
        )

if __name__ == "__main__":
    demo.launch()
