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

# Example usage
predictions = ["Normal chest x-ray", "Bilateral pleural effusions noted"]
ground_truth = ["Normal chest radiograph", "Small bilateral pleural effusions present"]

# Compute BLEU score
results = rrg_metric.compute(
    metric="bleu",
    preds=predictions,
    gts=ground_truth,
    per_sample=True,
    verbose=True
)

print(f"Total BLEU score: {results['total_results']}")
if results['per_sample_results']:
    print(f"Per-sample scores: {results['per_sample_results']}")
```

## Parameters
### `compute(metric, preds, gts, per_sample=False, verbose=False)`
#### Required Parameters:
- `metric` (str): The evaluation metric to use. Must be one of: ["bleu", "rouge", "meteor", "bertscore", "f1radgraph", "f1chexbert"]
- `preds` (List[str]): List of model predictions/generated texts
- `gts` (List[str]): List of ground truth/reference texts

#### Optional Parameters:
- `per_sample` (bool, default=False): If True, returns scores for each individual prediction-reference pair
- `verbose` (bool, default=False): If True, displays progress bars and loading messages

#### Returns:
Dictionary containing:

- `total_results`: Overall score for the metric
- `per_sample_results`: Individual scores if per_sample=True (None otherwise)
- Additional results for specific metrics (e.g., parsed graphs for f1radgraph)


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
