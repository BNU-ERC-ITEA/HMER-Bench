# Seeing Symbols, Missing Structure: A Real-World Handwritten Mathematical Expression Recognition Benchmark for Large Models

## Overview

HMER-Bench is a real-world handwritten mathematical expression recognition benchmark designed to evaluate whether large vision-language models can recover not only local symbols, but also global two-dimensional mathematical structures.

The benchmark contains **13,513 handwritten expressions** across **13 fine-grained categories**, covering long expressions, multi-line derivations, vertical arithmetic, short division, systems of equations, matrices and determinants, chemical equations, artifact-corrupted expressions, and reduction/cancellation expressions.

![HMER-Bench Overview](assets/HMER-Bench.png)

We also provide **Schema-Anchored Structure-Aware Reasoning (SASR)**, a training-free inference framework that first identifies the structural schema of a handwritten expression and then performs schema-constrained transcription.

![SASR Framework](assets/SASR.png)

## Dataset

HMER-Bench is available for **non-commercial academic research** upon request.

Please complete the [HMER-Bench Dataset Access Request Form](docs/HMER-Bench_Dataset_Access_Request_Form.docx), obtain the signature of your supervisor or research group leader, and send the signed form to **jiangsheng@mail.bnu.edu.cn** with the subject `HMER-Bench Dataset Access Request`.

After approval, the dataset download link and archive password will be provided by email.

Redistribution or sharing of the dataset with third parties is not permitted.

### Dataset Structure

After obtaining access, organize the dataset as follows:

```text
data/
└── hmerbench/
    ├── SSE/
    │   ├── images/
    │   └── labels_space.txt
    ├── LSE/
    ├── ...
    └── MAD/
```

## Quick Start

### 1. Install dependencies

~~~bash
pip install openai tqdm
~~~

### 2. Start a Qwen3-VL endpoint

Start an OpenAI-compatible inference server for Qwen3-VL. The default endpoint used by the script is http://127.0.0.1:22002/v1.

### 3. Run SASR inference

~~~bash
python src/sasr/inference_sasr.py \
  --mode SASR \
  --model-name /path/to/Qwen3-VL-8B-Instruct \
  --base-url http://127.0.0.1:22002/v1 \
  --api-key EMPTY
~~~

Raw responses are written to outputs/qwen3_vl_8b_sasr_sasr/<CATEGORY>/origin_answer.json.

### 4. Postprocess raw responses

The original SASR postprocessing protocol extracts the final \boxed{...} expression (falling back to the final $...$ span), builds its longest-match token dictionary from the predictions, and writes result.txt, labels.txt, and dictionary.txt.

~~~bash
python src/postprocess/postprocess_sasr.py \
  --prediction-root outputs/qwen3_vl_8b_sasr_sasr
~~~

### 5. Evaluate ExpRate

~~~bash
python src/eval/evaluate.py \
  --ground-truth-root data/hmerbench \
  --prediction-root outputs/qwen3_vl_8b_sasr_sasr \
  --output outputs/qwen3_vl_8b_sasr_sasr/evaluation_results.txt
~~~

The evaluator reports per-category and overall ExpRate, along with mean token accuracy.


## Citation

If you use HMER-Bench or SASR in your research, please cite our paper:

```bibtex
@inproceedings{
jiang2026seeing,
title={Seeing Symbols, Missing Structure: A Real-World Handwritten Mathematical Expression Recognition Benchmark for Large Models},
author={Sheng Jiang and Lin Zhu and Runrui Li and Mei Wang and Qiannan Zhu and Yaoyao Zhong and Hua Huang},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=OKVCjNqsjK}
}
```