import numpy as np
from tqdm.auto import tqdm
import os
from typing import List, Dict, Union, Tuple, Any, Optional, Literal
from sklearn.metrics import f1_score

def compute(
    metric: Literal["bleu", "rouge", "meteor", "bertscore", "f1radgraph", "chexbert", "ratescore"],
    preds: List[str],
    gts: List[str],
    per_sample: bool = False,
    verbose: bool = False,
    f1radgraph_model_type: Optional[Literal["radgraph", "radgraph-xl", "echograph"]] = "radgraph-xl",
    f1radgraph_reward_level: Optional[Literal["simple", "partial", "complete", "all"]] = "complete",
    cache_dir = None,
) -> Dict[str, Any]:
    """
    Compute evaluation metrics for radiology report generation.

    This function supports multiple evaluation metrics including BLEU, ROUGE, METEOR,
    BERTScore, F1RadGraph, and CheXbert scores (F1 CheXbert, SembScore). It can compute both aggregate and per-sample
    scores depending on the parameters.

    Args:
        metric (str): Evaluation metric to compute. Must be one of "bleu", "rouge",
            "meteor", "bertscore", "f1radgraph", "chexbert", or "ratescore".
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
        cache_dir (optional): Cache directory for huggingface model downloads.

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
    per_sample_results = None
    additional_results = {}
    assert len(preds) == len(gts), "Number of predictions and ground truths should be the same"

    iters = zip(preds, gts)
    log = lambda: None
    if verbose:
        if metric in ["f1radgraph", "chexbert", "ratescore"]:
            iters = zip(preds, gts)
            log = print
            print(f"Progress bar not available for '{metric}'.")
        else:
            iters = tqdm(iters, total=len(preds))
            log = iters.set_description

    if metric in ["bleu", "rouge", "meteor"]:
        from evaluate import load
        
        log(f"Loading '{metric}' computer...")
        computer = load(metric)

        key = "rougeL" if metric == "rouge" else metric

        log(f"Computing '{metric}' scores...")
        if per_sample:
            per_sample_results = []
            for pred, gt in iters:
                per_sample_result = computer.compute(
                    predictions=[pred],
                    references=[gt],
                )[key]
                per_sample_results.append(per_sample_result)
                total_results = np.mean(per_sample_results)
        else:
            total_results = computer.compute(
                predictions=preds,
                references=gts,
            )[key]

    elif metric == "bertscore":
        from evaluate import load
        
        log(f"Loading '{metric}' computer...")
        computer = load(metric)

        log(f"Computing '{metric}' scores...")
        per_sample_results = computer.compute(
            predictions=preds,
            references=gts,
            lang="en",
        )["f1"]
        total_results = np.mean(per_sample_results)

    elif metric == "f1radgraph":
        from .radgraph_gpu import F1RadGraph
        
        log(f"Loading '{metric}' computer...")
        log(f"Model type: {f1radgraph_model_type}, Reward level: {f1radgraph_reward_level}")
        computer = F1RadGraph(model_type=f1radgraph_model_type, reward_level=f1radgraph_reward_level)

        log(f"Computing '{metric}' scores...")
        total_results, per_sample_results, pred_graphs, gt_graphs = computer(hyps=preds, refs=gts)
        additional_results = {
            "pred_graphs": pred_graphs,
            "gt_graphs": gt_graphs
        }

    elif metric == "chexbert":
        from .chexbert import CheXbert
        
        # Will always return both scores: F1 CheXbert and SembScore.
        log(f"Loading '{metric}' computer...")
        computer = CheXbert(cache_dir=cache_dir)

        log(f"Computing '{metric}' scores...")
        accuracy, accuracy_not_averaged, class_report, class_report_5, sembscores = computer(hyps=preds, refs=gts)

        total_results = {
            "f1chexbert" : class_report_5['micro avg']['f1-score'],
            "sembscore"  : np.mean(sembscores)
        }
        per_sample_results = {
            "f1chexbert" : None,
            "sembscore"  : sembscores
        }
        additional_results = {
            "f1chexbert_accuracy"              : accuracy,
            "f1chexbert_accuracy_not_averaged" : accuracy_not_averaged,
            "f1chexbert_micro_precision_14"    : class_report['micro avg']['precision'],
            "f1chexbert_micro_recall_14"       : class_report['micro avg']['recall'],
            "f1chexbert_micro_f1_14"           : class_report['micro avg']['f1-score'],
            "f1chexbert_micro_precision_5"     : class_report_5['micro avg']['precision'],
            "f1chexbert_micro_recall_5"        : class_report_5['micro avg']['recall'],
            "f1chexbert_micro_f1_5"            : class_report_5['micro avg']['f1-score'],
            "f1chexbert_macro_precision_14"    : class_report['macro avg']['precision'],
            "f1chexbert_macro_recall_14"       : class_report['macro avg']['recall'],
            "f1chexbert_macro_f1_14"           : class_report['macro avg']['f1-score'],
            "f1chexbert_macro_precision_5"     : class_report_5['macro avg']['precision'],
            "f1chexbert_macro_recall_5"        : class_report_5['macro avg']['recall'],
            "f1chexbert_macro_f1_5"            : class_report_5['macro avg']['f1-score'],
        }

    elif metric == "ratescore":
        from RaTEScore import RaTEScore
        log(f"Loading '{metric}' computer...")
        computer = RaTEScore()

        log(f"Computing '{metric}' scores...")
        per_sample_results = computer.compute_score(preds, gts)

        total_results = np.mean(per_sample_results)        

    else:
        raise ValueError(f"Invalid metric: {metric}")

    log("Done.")

    return {
        "total_results": total_results,
        "per_sample_results": per_sample_results,
        **additional_results
    }
