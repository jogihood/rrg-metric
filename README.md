# rrg-metric

A Python package for evaluating Radiology Report Generation (RRG) using multiple metrics including:\
BLEU, ROUGE, METEOR, BERTScore, F1RadGraph, F1CheXbert, and SembScore.

## Features

- Multiple evaluation metrics supported:
  - BLEU
  - ROUGE
  - METEOR
  - BERTScore
  - F1 RadGraph
  - F1 CheXbert
  - SembScore (CheXbert vector similarity)
  - [RaTEScore](https://github.com/MAGIC-AI4Med/RaTEScore) (Entity-aware metric)
  - [GREEN](https://github.com/Stanford-AIMI/GREEN) (LLM-based metric)
- Easy-to-use API
- Support for batch processing
- Detailed per-sample and aggregated results
- Visualization tools for correlation analysis

## TODO
- Add CLI usage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jogihood/rrg-metric.git
cd rrg-metric
```

2. Install the required packages using pip:
```bash
pip install -r requirements.txt
```

## Usage

### Metric Computation

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

### Visualization (alpha)

The package provides visualization tools for correlation analysis between metric scores and radiologist error counts:

For preprocessing tools related to radiology error validation (ReXVal), please check: https://github.com/jogihood/rexval-preprocessor

```python
import rrg_metric
import matplotlib.pyplot as plt

# Example data
metric_scores = [0.8, 0.7, 0.9, 0.6, 0.85]
error_counts = [1, 2, 0, 3, 1]

# Create correlation plot
ax, tau, tau_ci = rrg_metric.plot_corr(
   metric="BLEU",
   metric_scores=metric_scores,
   radiologist_error_counts=error_counts,
   error_type="total",          # or "significant"
   color='blue',                # custom color
   scatter_alpha=0.6,           # scatter point transparency
   show_tau=True               # show Kendall's tau in title
)

print(f"Kendall's tau: {tau:.3f}")
print(f"95% CI: [{tau_ci[0]:.3f}, {tau_ci[1]:.3f}]")
plt.show()
```

## Parameters

### `compute(metric, preds, gts, per_sample=False, verbose=False)`
#### Required Parameters:
- `metric` (str): The evaluation metric to use. Must be one of: ["bleu", "rouge", "meteor", "bertscore", "f1radgraph", "chexbert", "ratescore"]
- `preds` (List[str]): List of model predictions/generated texts
- `gts` (List[str]): List of ground truth/reference texts

#### Optional Parameters:
- `per_sample` (bool, default=False): If True, returns scores for each individual prediction-reference pair
- `verbose` (bool, default=False): If True, displays progress bars and loading messages
- `f1radgraph_model_type` / `f1radgraph_reward_level`: Parameters for RadGraph. Recommend default values
- `cache_dir`: `cache_dir` for huggingface model downloads

### `plot_corr(metric, metric_scores, radiologist_error_counts, error_type="total", ax=None, **params)`
#### Required Parameters:
- `metric` (str): Name of the metric being visualized
- `metric_scores` (List[float]): List of metric scores
- `radiologist_error_counts` (List[float]): List of radiologist error counts

#### Optional Parameters:
- `error_type` (str, default="total"): Type of error to plot. Must be either "total" or "significant"
- `ax` (matplotlib.axes.Axes, default=None): Matplotlib axes for plotting. If None, creates new figure and axes
- Additional parameters for plot customization (see docstring for details)

## Requirements
- Python 3.10+
- Other dependencies listed in `requirements.txt`

## Contributing
This repository is still under active development. If you encounter any issues or bugs, I would really appreciate if you could submit a Pull Request. Your contributions will help make this package more robust and useful for the community!
