# cache

This folder is the cache directory for Hugging Face (HF).

When using online mode, downloaded models will be cached in this folder.

For [offline mode](https://huggingface.co/docs/transformers/main/installation#offline-mode) use, please download the models in advance and specify the model directory,
such as the `surya_det3` model below.

The folder structure for `./cache/huggingface/hub/models--vikp--surya_det3` is as follows.

```
.
├── blobs
│   ├── 17579df25d3b063dedb036aaca5b495efe5088b8
│   ├── 5a2a74e413345541b7ca0db0cb1d41785649eb99fe6a1b5166aa8bd7c0a8881d
│   ├── 6d76255a802ec614336406c974998559b5cae01b112b47ec7eab1ed39b5fdb4c
│   ├── 9888778190fd6ecff72bde2ecab5b24cda345851
│   ├── a6344aac8c09253b3b630fb776ae94478aa0275b
│   └── ac9b1b39818223e631ec9982956df6d65b33f082
├── refs
│   └── main
└── snapshots
    └── 467ee9ec33e6e6c5f73e57dbc1415b14032f5b95
        ├── config.json -> ../../blobs/9888778190fd6ecff72bde2ecab5b24cda345851
        ├── .gitattributes -> ../../blobs/a6344aac8c09253b3b630fb776ae94478aa0275b
        ├── model.safetensors -> ../../blobs/5a2a74e413345541b7ca0db0cb1d41785649eb99fe6a1b5166aa8bd7c0a8881d
        ├── preprocessor_config.json -> ../../blobs/17579df25d3b063dedb036aaca5b495efe5088b8
        ├── README.md -> ../../blobs/ac9b1b39818223e631ec9982956df6d65b33f082
        └── training_args.bin -> ../../blobs/6d76255a802ec614336406c974998559b5cae01b112b47ec7eab1ed39b5fdb4c

4 directories, 13 files
```

and `./cache/huggingface/hub/models--vikp--surya_rec2` like this

```
.
├── blobs
│   ├── 2f525ec0be1f2e8cb257a7b3e01de3bd003f0e81
│   ├── 31bdd446acbf8a47ea46d7d0a4998f145f0cc75a
│   ├── 5497e8690cfe93cbedec7efaf91f6ac734496ac8
│   ├── 93c190b5690dd55aac16723222a9909e2be0faec
│   ├── 9a75b64cbeaed0.8.1559bcda4e12c1235de62b5bce787d57cf56a9c3a7123d1
│   ├── a6344aac8c09253b3b630fb776ae94478aa0275b
│   ├── a83ef0a8114bd50cc650e08a9738c0f6345f5186
│   ├── dd34282c30833587a799d334d44a637694d41c8e
│   └── e237701f4293e736f74d2c968582935590107034
├── refs
│   └── main
└── snapshots
    └── 6611509b2c3a32c141703ce19adc899d9d0abf41
        ├── added_tokens.json -> ../../blobs/93c190b5690dd55aac16723222a9909e2be0faec
        ├── config.json -> ../../blobs/5497e8690cfe93cbedec7efaf91f6ac734496ac8
        ├── generation_config.json -> ../../blobs/e237701f4293e736f74d2c968582935590107034
        ├── .gitattributes -> ../../blobs/a6344aac8c09253b3b630fb776ae94478aa0275b
        ├── model.safetensors -> ../../blobs/9a75b64cbeaed0.8.1559bcda4e12c1235de62b5bce787d57cf56a9c3a7123d1
        ├── preprocessor_config.json -> ../../blobs/dd34282c30833587a799d334d44a637694d41c8e
        ├── README.md -> ../../blobs/a83ef0a8114bd50cc650e08a9738c0f6345f5186
        ├── special_tokens_map.json -> ../../blobs/2f525ec0be1f2e8cb257a7b3e01de3bd003f0e81
        └── tokenizer_config.json -> ../../blobs/31bdd446acbf8a47ea46d7d0a4998f145f0cc75a

4 directories, 19 files
```

and `./cache/huggingface/hub/models--vikp--surya_tablerec` like this

```
.
├── blobs
│   ├── 2f525ec0be1f2e8cb257a7b3e01de3bd003f0e81
│   ├── 4f269f7454bc6161d8482c86f1e02752da76bf36
│   ├── 59a0e54bb3f1ccc8313ed1a75035435e477773ecc62cc1bb0cee0e5dc58889c8
│   ├── 5f777357fc63326274fb93cbedf4948d61a89439
│   ├── 93c190b5690dd55aac16723222a9909e2be0faec
│   ├── a6344aac8c09253b3b630fb776ae94478aa0275b
│   ├── b1610cd7faa8eaa790a69059e2686534a95f4ef8
│   ├── bc2c740bf5bdf1af8a4909cb79c69c950745725b
│   └── f9cf6e019c3303cdbfdbae10cf9d4057b65ba050
├── refs
│   └── main
└── snapshots
    └── 8bca165f81e9cee5fb382413eb23175079917d14
        ├── added_tokens.json -> ../../blobs/93c190b5690dd55aac16723222a9909e2be0faec
        ├── config.json -> ../../blobs/5f777357fc63326274fb93cbedf4948d61a89439
        ├── generation_config.json -> ../../blobs/b1610cd7faa8eaa790a69059e2686534a95f4ef8
        ├── .gitattributes -> ../../blobs/a6344aac8c09253b3b630fb776ae94478aa0275b
        ├── model.safetensors -> ../../blobs/59a0e54bb3f1ccc8313ed1a75035435e477773ecc62cc1bb0cee0e5dc58889c8
        ├── preprocessor_config.json -> ../../blobs/bc2c740bf5bdf1af8a4909cb79c69c950745725b
        ├── README.md -> ../../blobs/4f269f7454bc6161d8482c86f1e02752da76bf36
        ├── special_tokens_map.json -> ../../blobs/2f525ec0be1f2e8cb257a7b3e01de3bd003f0e81
        └── tokenizer_config.json -> ../../blobs/f9cf6e019c3303cdbfdbae10cf9d4057b65ba050

4 directories, 19 files
```

and `./cache/huggingface/hubmodels--datalab-to--surya_layout0` like this


```
.
├── blobs
│   ├── 15a199bdf136c1ae9f4249308032f67bc0ef74e6770db2a87f55f72503914e00
│   ├── 6284a0fcdd550cdae76a90177664e8f34cf7a8cc
│   ├── 776774696986893ca5eb478899ea9d06c20435c5
│   ├── a6344aac8c09253b3b630fb776ae94478aa0275b
│   └── e389089e7f1b85048e6dc1a9f84b51982c04b9f0
├── refs
│   └── main
└── snapshots
    └── 421ac206a400227ea714d47a405e53ce74374957
        ├── config.json -> ../../blobs/e389089e7f1b85048e6dc1a9f84b51982c04b9f0
        ├── model.safetensors -> ../../blobs/15a199bdf136c1ae9f4249308032f67bc0ef74e6770db2a87f55f72503914e00
        ├── preprocessor_config.json -> ../../blobs/776774696986893ca5eb478899ea9d06c20435c5
        └── README.md -> ../../blobs/6284a0fcdd550cdae76a90177664e8f34cf7a8cc

4 directories, 10 files
```

and `./cache/huggingface/hubmodels--datalab-to--ocr_error_detection` like this


```
.
├── blobs
│   ├── 21f54a4b56685f29358f3a8de1f5b8d827357d07
│   ├── 9856c52ab99c8f7435bef6bf6e4c8a86a2594187
│   ├── 9bbecc17cabbcbd3112c14d6982b51403b264bfa
│   ├── a6344aac8c09253b3b630fb776ae94478aa0275b
│   ├── c305af17d2fcaf52c00b125a2dfabfbe16e71454
│   ├── cd3c57f2e967aad6a020decd1c1c41be-10
│   ├── e837bab60a5d204e29622d127c2dafe508aa0731
│   └── f4a46fa248690b0b2adc680e845ec8fd491eb24c
├── refs
│   └── main
└── snapshots
    └── c1cbda3757670fd520553eaa5197656d331de414
        ├── config.json -> ../../blobs/9856c52ab99c8f7435bef6bf6e4c8a86a2594187
        ├── model.safetensors -> ../../blobs/cd3c57f2e967aad6a020decd1c1c41be-10
        ├── README.md -> ../../blobs/c305af17d2fcaf52c00b125a2dfabfbe16e71454
        ├── special_tokens_map.json -> ../../blobs/9bbecc17cabbcbd3112c14d6982b51403b264bfa
        ├── tokenizer_config.json -> ../../blobs/f4a46fa248690b0b2adc680e845ec8fd491eb24c
        ├── tokenizer.json -> ../../blobs/21f54a4b56685f29358f3a8de1f5b8d827357d07
        └── vocab.txt -> ../../blobs/e837bab60a5d204e29622d127c2dafe508aa0731

5 directories, 16 files
```

It will use
- `./cache/huggingface/hub/models--vikp--surya_det3/snapshots/467ee9ec33e6e6c5f73e57dbc1415b14032f5b95`
- `./cache/huggingface/hub/models--vikp--surya_rec2/snapshots/6611509b2c3a32c141703ce19adc899d9d0abf41`
- `./cache/huggingface/hub/models--vikp--surya_tablerec/snapshots/8bca165f81e9cee5fb382413eb23175079917d14`
- `./cache/huggingface/hub/hubmodels--datalab-to--surya_layout0/snapshots/421ac206a400227ea714d47a405e53ce74374957`
- `./cache/huggingface/hub/hubmodels--datalab-to--ocr_error_detection/snapshots/c1cbda3757670fd520553eaa5197656d331de414`

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
huggingface-cli download vikp/surya_rec2 --repo-type model --revision main --cache-dir ./cache/huggingface/hub
huggingface-cli download vikp/surya_tablerec --repo-type model --revision main --cache-dir ./cache/huggingface/hub
huggingface-cli download datalab-to/surya_layout0 --revision main --cache-dir ./cache/huggingface/hub
huggingface-cli download datalab-to/ocr_error_detection --revision main --cache-dir ./cache/huggingface/hub
```