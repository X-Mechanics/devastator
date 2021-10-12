# Devastator: Multi-Modal Transformers

## Introduction

Devastator makes it easy to build high-performance and efficient transformers (with memory) on a wide variety of
data types.

## Blur: Transformers for text data

### Reproducing [Wikitext-103 benchmarks](https://paperswithcode.com/sota/language-modelling-on-wikitext-103)

- Download the wikitext-103 dataset

```bash
user@device$ cd download_path/devastator/
user@device:/devastator$ ./get_data_wt103.sh
```

- To reproduce the result from [Transformer-XL](http://arxiv.org/abs/1901.02860) (ppl ~ 24)
  
```bash
user@device:/devastator$ cd blur
user@device:/devastator/blur$ ./run_train_wt103_base.sh train
```


- To reproduce the result from [FNetAR](http://arxiv.org/abs/2107.10932) (ppl ~ 25)
  
```bash
user@device:/devastator$ cd blur
user@device:/devastator/blur$ ./run_train_wt103_base_fnetar.sh train
```

- To reproduce the result from [Feedback Transformer](http://arxiv.org/abs/2002.09402) (ppl < 21)
  
```bash
user@device:/devastator$ cd blur
user@device:/devastator/blur$ ./run_train_wt103_base_feedback.sh train
```

## Perceptor: Transformers for image and video data (NotImplemented)

## Blaster: Transformers for audio and wave-form data (NotImplemented)
 