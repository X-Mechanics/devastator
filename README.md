# Devastator: Multi-Modal Transformers

## Introduction

Devastator makes it easy to build high-performance and efficient transformers (with memory) on a wide variety of
data types.

- Blur: Transformers for text data

- Perceptor: Transformers for image and video data (NotImplemented)

- Blaster: Transformers for audio and wave-form data (NotImplemented)

## Quickstart

### Reproduce a perplexity score PPL~28.0 on `wikitext-103`

- Download the wikitext-103 dataset

```bash
user@device$ ./get_blur_data.sh
```

- Run training script

```bash
user@device$ python run_train_blur.py
```

- Compare with benchmarks: https://paperswithcode.com/sota/language-modelling-on-wikitext-103