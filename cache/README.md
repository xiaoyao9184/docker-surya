# cache

This folder is the cache directory for Hugging Face (HF).

When using online mode, downloaded models will be cached in this folder.

For [offline mode](https://huggingface.co/docs/transformers/main/installation#offline-mode) use, please download the models in advance and specify the model directory,
such as the `surya_order` model below.

The folder structure for `./cache/huggingface/hub/models--vikp--surya_order` is as follows.

```
├── blobs
│   ├── 4566f6f099f2c8ff7426500ff4a7d84a7a8514a0
│   ├── bb3f109ce8d6999ebfda8fbe7940ce7e68ea7e60
│   ├── cda4db4b7f886de50399b061100af28767cf2b71
│   └── d6688651c37649cc60492f18b5e6bf085f852e0f05e0543b6ca57f25fa884b4a
├── refs
│   └── main
└── snapshots
    └── 7b727d0a2c942cdc8596d186115f65c12c812bd8
        ├── config.json
        ├── generation_config.json
        ├── model.safetensors
        ├── preprocessor_config.json
        └── README.md
```

It will use `./cache/huggingface/hub/models--vikp--surya_order/snapshots/7b727d0a2c942cdc8596d186115f65c12c812bd8`.

For more details, refer to [up@cpu-offline/docker-compose.yml](./../docker/up@cpu-offline/docker-compose.yml).


## Pre-download for offline mode

Running in online mode will automatically download the model.

install cli

```bash
pip install -U "huggingface_hub[cli]"
```

download model

```bash
huggingface-cli download vikp/surya_det3 --repo-type model --revision main --cache-dir ./cache/huggingface/hub
huggingface-cli download vikp/surya_layout3 --repo-type model --revision main --cache-dir ./cache/huggingface/hub
huggingface-cli download vikp/surya_order --repo-type model --revision main --cache-dir ./cache/huggingface/hub
huggingface-cli download vikp/surya_rec2 --repo-type model --revision main --cache-dir ./cache/huggingface/hub
huggingface-cli download vikp/surya_tablerec --repo-type model --revision main --cache-dir ./cache/huggingface/hub
```