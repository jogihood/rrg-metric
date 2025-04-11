import torch
import os
import logging
import numpy as np
import torch.nn as nn
import pandas as pd
import warnings

from transformers import BertModel, AutoModel, AutoConfig, BertTokenizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics._classification import _check_targets
from huggingface_hub import hf_hub_download, list_repo_files
from sklearn.utils.sparsefuncs import count_nonzero

from .model import load_chexbert

warnings.filterwarnings("ignore")
logging.getLogger("urllib3").setLevel(logging.ERROR)

def generate_attention_masks(batch, source_lengths, device):
    """Generate masks for padded batches to avoid self-attention over pad tokens
    @param batch (Tensor): tensor of token indices of shape (batch_size, max_len)
                           where max_len is length of longest sequence in the batch
    @param source_lengths (List[Int]): List of actual lengths for each of the
                           sequences in the batch
    @param device (torch.device): device on which data should be

    @returns masks (Tensor): Tensor of masks of shape (batch_size, max_len)
    """
    masks = torch.ones(batch.size(0), batch.size(1), dtype=torch.float)
    for idx, src_len in enumerate(source_lengths):
        masks[idx, src_len:] = 0
    return masks.to(device)

def tokenize(impressions, tokenizer):
    imp = impressions.str.strip()
    imp = imp.replace('\n', ' ', regex=True)
    imp = imp.replace(r'\s+', ' ', regex=True)
    impressions = imp.str.strip()
    new_impressions = []
    for i in (range(impressions.shape[0])):
        tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
        if tokenized_imp:  # not an empty report
            res = tokenizer.encode_plus(tokenized_imp)['input_ids']
            if len(res) > 512:  # length exceeds maximum size
                # print("report length bigger than 512")
                res = res[:511] + [tokenizer.sep_token_id]
            new_impressions.append(res)
        else:  # an empty report
            new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
    return new_impressions

class CheXbert(nn.Module):
    def __init__(self, refs_filename=None, hyps_filename=None, device=None, cache_dir=None, **kwargs):
        super(CheXbert, self).__init__()
        self.refs_filename = refs_filename
        self.hyps_filename = hyps_filename

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device(device)

        # Model and tok
        self.model, self.tokenizer = load_chexbert(device=self.device, cache_dir=cache_dir)

        # Defining classes
        self.target_names = [
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
            "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices", "No Finding"]

        self.target_names_5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
        self.target_names_5_index = np.where(np.isin(self.target_names, self.target_names_5))[0]

    def get_label(self, report, mode="rrg"):
        impressions = pd.Series([report])
        out = tokenize(impressions, self.tokenizer)
        batch = torch.LongTensor([o for o in out])
        src_len = [b.shape[0] for b in batch]
        attn_mask = generate_attention_masks(batch, src_len, self.device)
        embeds, logits = self.model(batch.to(self.device), attn_mask)
        out = [logits[j].argmax(dim=1).item() for j in range(len(logits))]
        v = []
        if mode == "rrg":
            for c in out:
                if c == 0:
                    v.append('')
                if c == 3:
                    v.append(1)
                if c == 2:
                    v.append(0)
                if c == 1:
                    v.append(1)
            v = [1 if (isinstance(l, int) and l > 0) else 0 for l in v]

        elif mode == "classification":
            # https://github.com/stanfordmlgroup/CheXbert/blob/master/src/label.py#L124
            for c in out:
                if c == 0:
                    v.append('')
                if c == 3:
                    v.append(-1)
                if c == 2:
                    v.append(0)
                if c == 1:
                    v.append(1)
        else:
            raise NotImplementedError(mode)

        return v, embeds

    def forward(self, hyps, refs):
        if self.refs_filename is None:
            refs_chexbert = [self.get_label(l.strip()) for l in refs]
        else:
            if os.path.exists(self.refs_filename):
                refs_chexbert = [eval(l.strip()) for l in open(self.refs_filename).readlines()]
            else:
                refs_chexbert = [self.get_label(l.strip()) for l in refs]
                open(self.refs_filename, 'w').write('\n'.join(map(str, refs_chexbert)))

        hyps_chexbert = [self.get_label(l.strip()) for l in hyps]
        if self.hyps_filename is not None:
            open(self.hyps_filename, 'w').write('\n'.join(map(str, hyps_chexbert)))

        refs_chexbert, list_label_embeds = [r[0] for r in refs_chexbert], [r[1] for r in refs_chexbert]
        hyps_chexbert, list_pred_embeds = [h[0] for h in hyps_chexbert], [h[1] for h in hyps_chexbert]

        ########## SembScore ##########            
        np_label_embeds = torch.stack(list_label_embeds, dim=0).detach().cpu().numpy()
        np_pred_embeds = torch.stack(list_pred_embeds, dim=0).detach().cpu().numpy()
        
        sembscores = []
        for i, (label, pred) in enumerate(zip(np_label_embeds, np_pred_embeds)):
            sim_scores = (label * pred).sum() / (
                np.linalg.norm(label) * np.linalg.norm(pred))
            sembscores.append(sim_scores)

        ########## F1 CheXbert ##########
        refs_chexbert_5 = [np.array(r)[self.target_names_5_index] for r in refs_chexbert]
        hyps_chexbert_5 = [np.array(h)[self.target_names_5_index] for h in hyps_chexbert]

        # Accuracy
        accuracy = accuracy_score(y_true=refs_chexbert_5, y_pred=hyps_chexbert_5)
        # Per element accuracy
        y_type, y_true, y_pred = _check_targets(refs_chexbert_5, hyps_chexbert_5)
        differing_labels = count_nonzero(y_true - y_pred, axis=1)
        pe_accuracy = (differing_labels == 0).astype(np.float32)

        cr = classification_report(refs_chexbert, hyps_chexbert, target_names=self.target_names, output_dict=True)
        cr_5 = classification_report(refs_chexbert_5, hyps_chexbert_5, target_names=self.target_names_5,
                                     output_dict=True)

        return accuracy, pe_accuracy, cr, cr_5, sembscores

    def train(self, mode: bool = True):
        raise NotImplementedError
        # mode = False  # force False
        # self.training = mode
        # for module in self.children():
        #     module.train(mode)
        # return self