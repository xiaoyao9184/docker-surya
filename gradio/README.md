---
title: Surya
emoji: 🌍
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.8.0
python_version: '3.11'
app_file: space.py
pinned: false
license: apache-2.0
short_description: Gradio implementation of the Surya OCR
models:
  - vikp/surya_det3
  - vikp/surya_rec2
  - vikp/surya_tablerec
  - datalab-to/surya_layout
  - datalab-to/surya_tablerec
  - datalab-to/texify
  - datalab-to/ocr_error_detection
  - datalab-to/inline_math_det0
  - datalab-to/line_detector0
  - xiaoyao9184/surya_text_detection
  - xiaoyao9184/surya_text_recognition
  - xiaoyao9184/surya_table_recognition
  - xiaoyao9184/surya_texify
  - xiaoyao9184/surya_layout
  - xiaoyao9184/surya_ocr_error_detection
  - xiaoyao9184/surya_inline_math_detection
  - datalab-to/surya-ocr-2
  - datalab-to/surya-ocr-2-gguf
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Several environment variables need to be set when using hf space.

This will automatically download the pre-compiled llama.cpp as the backend for running the datalab-to/surya-ocr-2 model.

- SURYA_INFERENCE_BACKEND=llamacpp Required
- LLAMA_CPP_BINARY=llama-server Required
- LLAMA_CPP_RELEASE_URL=[https://github.com/ggml-org/llama.cpp/releases/download/b10091/llama-b10091-bin-ubuntu-x64.tar.gz](https://github.com/ggml-org/llama.cpp/releases/download/b10091/llama-b10091-bin-ubuntu-x64.tar.gz) Optional
- LLAMA_CPP_INSTALL_DIR=~/.cache/llama.cpp Optional
