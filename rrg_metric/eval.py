import numpy as np
from tqdm import tqdm
import os
from typing import List, Dict, Union, Tuple, Any, Optional, Literal

from sklearn.metrics import f1_score

from evaluate import load
from radgraph import F1RadGraph
from f1chexbert import F1CheXbert
from huggingface_hub import hf_hub_download

from appdirs import user_cache_dir

def compute(
    metric: Literal["bleu", "rouge", "meteor", "bertscore", "f1radgraph", "f1chexbert"],
    preds: List[str],
    gts: List[str],
    per_sample: bool = False,
    verbose: bool = False,
    f1radgraph_model_type: Optional[Literal["radgraph", "radgraph-xl", "echograph"]] = "radgraph-xl",
    f1radgraph_reward_level: Optional[Literal["simple", "partial", "complete", "all"]] = "complete",
) -> Dict[str, Any]:
    """
    Compute evaluation metrics for radiology report generation.

    This function supports multiple evaluation metrics including BLEU, ROUGE, METEOR,
    BERTScore, F1RadGraph, and F1CheXbert. It can compute both aggregate and per-sample
    scores depending on the parameters.

    Args:
        metric (str): Evaluation metric to compute. Must be one of "bleu", "rouge",
            "meteor", "bertscore", "f1radgraph", or "f1chexbert".
        preds (List[str]): List of model predictions/generated texts
        gts (List[str]): List of ground truth/reference texts
        per_sample (bool, optional): If True, returns scores for each individual 
            prediction-reference pair. Defaults to False.
        verbose (bool, optional): If True, displays progress bars and loading 
            messages. Defaults to False.
        f1radgraph_model_type (str, optional): Model type for F1RadGraph. Must be one of
            "radgraph", "radgraph-xl", or "echograph". Defaults to "radgraph".
        f1radgraph_reward_level (str, optional): Reward level for F1RadGraph. Must be one of
            "simple", "partial", "complete", or "all". Defaults to "all".

    Returns:
        Dict[str, Any]: A dictionary containing evaluation results:
            - total_results: Overall score for the metric
            - per_sample_results: Individual scores if per_sample=True (None otherwise)
            - Additional metric-specific results (e.g., parsed graphs for f1radgraph)

    Raises:
        ValueError: If the specified metric is not supported or if the number of 
            predictions and ground truths don't match.
        AssertionError: If the lengths of preds and gts lists don't match.
    
    Examples:
        >>> preds = ["Normal chest x-ray", "Bilateral pleural effusions noted"]
        >>> gts = ["Normal chest radiograph", "Bilateral effusions present"]
        >>> result = compute("bleu", preds=preds, gts=gts)
        >>> print(f"BLEU score: {result['total_results']}")

        >>> # Compute per-sample scores
        >>> result = compute("bertscore", preds=preds, gts=gts, per_sample=True)
        >>> print(f"Individual BERTScores: {result['per_sample_results']}")
    """
    additional_results = {}
    assert len(preds) == len(gts), "Number of predictions and ground truths should be the same"
    iters = tqdm((zip(preds, gts)), total=len(preds)) if verbose else zip(preds, gts)
    log = iters.set_description if verbose else None
    # TODO: Modify progress bar

    if metric in ["bleu", "rouge", "meteor", "bertscore"]:
        if verbose: log(f"Loading '{metric}' computer...")
        computer = load(metric)

        if verbose: log(f"Computing '{metric}' scores...")
        k = metric
        if metric == "rouge": k = "rougeL"
        if metric == "bertscore": k = "f1"

        params = {} if metric != "bertscore" else {"lang": "en"}
        if per_sample:
            per_sample_results = []
            for pred, gt in iters:
                params["predictions"] = [pred]
                params["references"]  = [gt]
                per_sample_results.append(computer.compute(**params)[k])
                total_results = np.mean(per_sample_results)
        else:
            params["predictions"] = preds
            params["references"]  = gts
            per_sample_results = None
            total_results = computer.compute(**params)[k]

    elif metric == "f1radgraph":
        if verbose: log(f"Loading '{metric}' computer...")
        if verbose: log(f"Model type: {f1radgraph_model_type}, Reward level: {f1radgraph_reward_level}")
        computer = F1RadGraph(model_type=f1radgraph_model_type, reward_level=f1radgraph_reward_level)

        if verbose: log(f"Computing '{metric}' scores... Progress bar not available for '{metric}'")
        total_results, per_sample_results, pred_graphs, gt_graphs = computer(hyps=preds, refs=gts)
        additional_results = {
            "pred_graphs": pred_graphs,
            "gt_graphs": gt_graphs
        }

    elif metric == "f1chexbert":
        if verbose: log(f"Loading '{metric}' computer...")
        
        chexbert_cache_dir = user_cache_dir("chexbert")
        os.makedirs(chexbert_cache_dir, exist_ok=True)
        chexbert_checkpoint = os.path.join(chexbert_cache_dir, "chexbert.pth")

        # TODO: modifiable cache_dir
        if not os.path.exists(chexbert_checkpoint):
            if verbose: log(f"'{metric}' model not found. Downloading...")
            chexbert_checkpoint_cache = hf_hub_download(
                repo_id='StanfordAIMI/RRG_scorers',
                cache_dir=chexbert_cache_dir,
                filename="chexbert.pth"
            )
            os.symlink(chexbert_checkpoint_cache, chexbert_checkpoint)

        computer = F1CheXbert()

        if verbose: log(f"Computing '{metric}' scores...")
        # hyps = [computer.get_label(pred) for pred in preds]
        # refs = [computer.get_label(gt) for gt in gts]

        # per_sample_results = [f1_score([ref], [hyp], average='micro') for ref, hyp in iters]
        # total_results = np.mean(per_sample_results)

        per_sample_results = None
        total_results = computer(hyps=preds, refs=gts)

    else:
        raise ValueError(f"Invalid metric: {metric}")

    return {
        "total_results": total_results,
        "per_sample_results": per_sample_results,
        **additional_results
    }


# if __name__ == '__main__':
#     COMPUTERS = {
#         "BLEU":         "load('bleu')",
#         "ROUGE":        "load('rouge')",
#         "METEOR":       "load('meteor')",
#         "BERTScore":    "load('bertscore')",
#         "F1RadGraph":   "F1RadGraph(reward_level=RADGRAPH_REWARD_LEVEL)",
#         "F1CheXbert":   "F1CheXbert()",
#     }
#     computer = None

#     if args.metric not in COMPUTERS:
#         raise ValueError(f"Invalid metric: {args.metric}")
