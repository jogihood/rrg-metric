import numpy as np
from tqdm import tqdm
import os

from sklearn.metrics import f1_score

from evaluate import load
from radgraph import F1RadGraph
from f1chexbert import F1CheXbert
from huggingface_hub import hf_hub_download

from appdirs import user_cache_dir

from .config import (
    RADGRAPH_REWARD_LEVEL,
)

# parser = argparse.ArgumentParser(description='Evaluate the model')
# parser.add_argument('--input', '-i', type=str, required=True, help='Input file')
# parser.add_argument('--output', '-o', type=str, required=True, help='Output file')
# parser.add_argument('--metric', '-m' type=str, default='bleu', help='Evaluation metric')
# args = parser.parse_args()


def compute(metric, preds, gts, per_sample=False, verbose=False):
    additional_results = {}
    assert len(preds) == len(gts), "Number of predictions and ground truths should be the same"
    iters = tqdm((zip(preds, gts)), total=len(preds)) if verbose else zip(preds, gts)
    log = iters.set_description
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
        computer = F1RadGraph(reward_level=RADGRAPH_REWARD_LEVEL)

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
