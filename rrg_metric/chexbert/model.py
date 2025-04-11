import torch
import torch.nn as nn

from collections import OrderedDict
from transformers import BertModel, AutoModel, AutoConfig, BertTokenizer
from huggingface_hub import hf_hub_download, list_repo_files

class bert_encoder(nn.Module):
    def __init__(self, return_logits=False, p=0.1, clinical=False, freeze_embeddings=False, 
                 pretrain_path=None, inference=False, cache_dir=None):
        """ Unified BERT encoder/labeler module
        @param return_logits (boolean): If True, return logits from linear heads; if False, return CLS embedding
        @param p (float): p to use for dropout in the linear heads, 0.1 by default is consistent with
                          transformers.BertForSequenceClassification
        @param clinical (boolean): True if Bio_Clinical BERT desired, False otherwise. Ignored if
                                   pretrain_path is not None
        @param freeze_embeddings (boolean): true to freeze bert embeddings during training
        @param pretrain_path (string): path to load checkpoint from
        @param inference (boolean): if True, use BERT config instead of pretrained weights
        """
        super(bert_encoder, self).__init__()
        
        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path, cache_dir=cache_dir)
        elif clinical:
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", cache_dir=cache_dir)
        elif inference:
            config = AutoConfig.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
            self.bert = AutoModel.from_config(config)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
            
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
                
        self.return_logits = return_logits
        self.dropout = nn.Dropout(p)
        # size of the output of transformer's last layer
        hidden_size = self.bert.pooler.dense.in_features
        # classes: present, absent, unknown, blank for 12 conditions + support devices
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        # classes: yes, no for the 'no finding' observation
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))
        
    def forward(self, source_padded, attention_mask):
        """ Forward pass of the unified model
        @param source_padded (torch.LongTensor): Tensor of word indices with padding, shape (batch_size, max_len)
        @param attention_mask (torch.Tensor): Mask to avoid attention on padding tokens, shape (batch_size, max_len)
        @returns out: 
            If logits=True: A list of size 14 containing tensors. The first 13 have shape
                           (batch_size, 4) and the last has shape (batch_size, 2)
            If logits=False: CLS hidden representation with shape (batch_size, hidden_size)
        """
        # shape (batch_size, max_len, hidden_size)
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        # shape (batch_size, hidden_size)
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        
        # Apply dropout only for logits calculation
        if self.return_logits:
            cls_hidden_with_dropout = self.dropout(cls_hidden)
            logits = []
            for i in range(14):
                logits.append(self.linear_heads[i](cls_hidden_with_dropout))
            return cls_hidden, logits
        else:
            return cls_hidden, None

def load_chexbert(checkpoint=None, device=None, cache_dir=None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = bert_encoder(return_logits=True, inference=True)
    
    # Downloading pretrained model from huggingface
    if checkpoint is None:
        checkpoint = hf_hub_download(repo_id='StanfordAIMI/RRG_scorers', filename='chexbert.pth', cache_dir=cache_dir)

    # Load model
    state_dict = torch.load(checkpoint, map_location=device)['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove `module.`
        new_state_dict[name] = v

    # Load params
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model = model.eval()

    for name, param in model.named_parameters():
        param.requires_grad = False

    return model, tokenizer