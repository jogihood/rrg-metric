# rrg-metric

A Python package for evaluating Radiology Report Generation (RRG) using multiple metrics including:\
BLEU, ROUGE, METEOR, BERTScore, F1RadGraph, and F1CheXbert.

## Features

- Multiple evaluation metrics supported:
  - BLEU (Bilingual Evaluation Understudy)
  - ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
  - METEOR (Metric for Evaluation of Translation with Explicit ORdering)
  - BERTScore
  - F1 RadGraph (Specialized for radiology report graphs)
  - F1 CheXbert (Specialized for chest X-ray reports)
- Easy-to-use API
- Support for batch processing
- Detailed per-sample and aggregated results

## TODO
- Add SembScore (CheXbert Vector Similarity)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jogihood/rrg-metric.git
cd rrg-metric
```

2. Create and activate a conda environment using the provided `environment.yml`:
```bash
conda env create -f environment.yml
conda activate rrg-metric
```

Alternatively, you can install the required packages using pip:
```bash
pip install -r requirements.txt
```

## Usage

Here's a simple example of how to use the package:

```python
import rrg_metric

# Example predictions and ground truths
predictions = [
    "no acute cardiopulmonary abnormality",
    "et tube terminates 2 cm above the carina retraction by several centimeters is recommended"
]
ground_truths = [
    "no acute cardiopulmonary abnormality",
    "endotracheal tube terminates 2 5 cm above the carina"
]

# Compute metrics
bleu_result = rrg_metric.compute("bleu", preds=predictions, gts=ground_truths)
rouge_result = rrg_metric.compute("rouge", preds=predictions, gts=ground_truths)
bertscore_result = rrg_metric.compute("bertscore", preds=predictions, gts=ground_truths)
f1radgraph_result = rrg_metric.compute("f1radgraph", preds=predictions, gts=ground_truths)
f1chexbert_result = rrg_metric.compute("f1chexbert", preds=predictions, gts=ground_truths)

# Access results
print(f"BLEU Score: {bleu_result['total_results']}")
```

## Available Metrics

The package supports the following metrics:

1. `bleu`: Basic BLEU score computation
2. `rouge`: ROUGE-L score for evaluating summary quality
3. `meteor`: METEOR score for machine translation evaluation
4. `bertscore`: Contextual embedding-based evaluation using BERT
5. `f1radgraph`: Specialized metric for evaluating radiology report graphs
6. `f1chexbert`: Specialized metric for chest X-ray report evaluation

You can check available metrics using:
```python
print(rrg_metric.AVAILABLE_METRICS)
```

## Return Format

Each metric computation returns a dictionary containing:
- `total_results`: Aggregated score for the entire batch
- `per_sample_results`: Individual scores for each prediction-reference pair
- Additional metric-specific results (e.g., graphs for F1RadGraph)

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- Evaluate
- RadGraph
- F1CheXbert
- Other dependencies listed in `requirements.txt`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
