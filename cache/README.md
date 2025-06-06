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

and `./cache/huggingface/hub/models--datalab-to--surya_tablerec` like this

```
./cache/huggingface/hub/models--datalab-to--surya_tablerec
├── blobs
│   ├── 38da0234ebdf63bed033dba22ed1005e0734b9f0
│   ├── 3c7c6a10d0d6f251612d9fbc86faab06d66fd918
│   ├── 8e5093c424a4c8b98d153519f5240532388e209158f656ee701989174dcad6c4
│   ├── a6344aac8c09253b3b630fb776ae94478aa0275b
│   └── e6c9b7e8b7850a7c02ad3895985f3e86d69256d5
├── refs
│   └── main
└── snapshots
    └── 7327dac38c300b2f6cd0501ebc2347dd3ef7fcf2
        ├── config.json -> ../../blobs/38da0234ebdf63bed033dba22ed1005e0734b9f0
        ├── model.safetensors -> ../../blobs/8e5093c424a4c8b98d153519f5240532388e209158f656ee701989174dcad6c4
        ├── preprocessor_config.json -> ../../blobs/e6c9b7e8b7850a7c02ad3895985f3e86d69256d5
        └── README.md -> ../../blobs/3c7c6a10d0d6f251612d9fbc86faab06d66fd918

5 directories, 10 files
```

and `./cache/huggingface/hub/models--datalab-to--surya_layout` like this


```
.
├── blobs
│   ├── 3890b5c4361d8c355b18efcbc083d80b-10
│   ├── 5551e790c7d99f076fb8ff17b38339138e4fc543
│   ├── 776774696986893ca5eb478899ea9d06c20435c5
│   ├── a6344aac8c09253b3b630fb776ae94478aa0275b
│   └── c305af17d2fcaf52c00b125a2dfabfbe16e71454
├── refs
│   └── main
└── snapshots
    └── 7ac8e390226ee5fa2125dd303d827f79d31d1a1f
        ├── config.json -> ../../blobs/5551e790c7d99f076fb8ff17b38339138e4fc543
        ├── model.safetensors -> ../../blobs/3890b5c4361d8c355b18efcbc083d80b-10
        ├── preprocessor_config.json -> ../../blobs/776774696986893ca5eb478899ea9d06c20435c5
        └── README.md -> ../../blobs/c305af17d2fcaf52c00b125a2dfabfbe16e71454

5 directories, 10 files
```

and `./cache/huggingface/hub/models--datalab-to--ocr_error_detection` like this


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

and `./cache/huggingface/hub/models--datalab-to--texify` like this


```
.
├── blobs
│   ├── 0b7eef4d4e38242a8bfe86b25350ee57a80ddea2-gzip
│   ├── 1c7a55edfe48b2b661b042bf711fbfce-10
│   ├── 5e12d9ba900b47f9ffa293b89737baa294699859
│   ├── 80cf0856476a03804ea7139a7ded72575f3f38c6-gzip
│   ├── a6344aac8c09253b3b630fb776ae94478aa0275b-gzip
│   ├── a782b2f1cdab4d0bacb2dc0f85d02c4b1e31f0bd-gzip
│   ├── e3cd24912053c61b0331779b976df959-10
│   └── fbed506ad438e179c130257fabd67cb49c932793-gzip
├── refs
│   └── main
└── snapshots
    └── 8f1d761762b3e977e9e62cebfca487d489556abc
        ├── config.json -> ../../blobs/80cf0856476a03804ea7139a7ded72575f3f38c6-gzip
        ├── model.safetensors -> ../../blobs/e3cd24912053c61b0331779b976df959-10
        ├── preprocessor_config.json -> ../../blobs/0b7eef4d4e38242a8bfe86b25350ee57a80ddea2-gzip
        ├── README.md -> ../../blobs/5e12d9ba900b47f9ffa293b89737baa294699859
        ├── special_tokens_map.json -> ../../blobs/a782b2f1cdab4d0bacb2dc0f85d02c4b1e31f0bd-gzip
        ├── tokenizer.json -> ../../blobs/fbed506ad438e179c130257fabd67cb49c932793-gzip
        └── training_args.bin -> ../../blobs/1c7a55edfe48b2b661b042bf711fbfce-10

5 directories, 16 files
```

It will use
- `./cache/huggingface/hub/models--vikp--surya_det3/snapshots/467ee9ec33e6e6c5f73e57dbc1415b14032f5b95`
- `./cache/huggingface/hub/models--vikp--surya_rec2/snapshots/6611509b2c3a32c141703ce19adc899d9d0abf41`
- `./cache/huggingface/hub/models--datalab-to--surya_tablerec/snapshots/7327dac38c300b2f6cd0501ebc2347dd3ef7fcf2`
- `./cache/huggingface/hub/models--datalab-to--surya_layout/snapshots/7ac8e390226ee5fa2125dd303d827f79d31d1a1f`
- `./cache/huggingface/hub/models--datalab-to--ocr_error_detection/snapshots/c1cbda3757670fd520553eaa5197656d331de414`
- `./cache/huggingface/hub/models--datalab-to--texify/snapshots/8f1d761762b3e977e9e62cebfca487d489556abc`

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
huggingface-cli download datalab-to/surya_tablerec --repo-type model --revision main --cache-dir ./cache/huggingface/hub
huggingface-cli download datalab-to/surya_layout --revision main --cache-dir ./cache/huggingface/hub
huggingface-cli download datalab-to/ocr_error_detection --revision main --cache-dir ./cache/huggingface/hub
huggingface-cli download datalab-to/texify --revision main --cache-dir ./cache/huggingface/hub
```