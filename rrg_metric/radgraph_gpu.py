import logging
import torch.nn as nn
import numpy as np

import json
import os
import tarfile
import torch
import importlib.metadata
from dotmap import DotMap

from radgraph.allennlp.data import Vocabulary
from radgraph.allennlp.data.dataset_readers import AllennlpDataset
from radgraph.allennlp.data.dataloader import PyTorchDataLoader
from radgraph.allennlp.data import token_indexers
from radgraph.allennlp.modules import token_embedders, text_field_embedders
from radgraph.allennlp.common.params import Params
from radgraph.dygie.data.dataset_readers.dygie import DyGIEReader
from radgraph.dygie.models import dygie

from radgraph.utils import (
    download_model,
    preprocess_reports,
    postprocess_reports,
    batch_to_device,
)

from appdirs import user_cache_dir

from radgraph import RadGraph as RadGraphCpu
from radgraph import F1RadGraph as F1RadGraphCpu

logging.getLogger("radgraph").setLevel(logging.CRITICAL)
logging.getLogger("allennlp").setLevel(logging.CRITICAL)

MODEL_MAPPING = {
    "radgraph": "radgraph.tar.gz",
    "radgraph-xl": "radgraph-xl.tar.gz",
    "echograph": "echograph.tar.gz",
}

version = importlib.metadata.version('radgraph')
CACHE_DIR = user_cache_dir("radgraph")
CACHE_DIR = os.path.join(CACHE_DIR, version)

class RadGraph(RadGraphCpu):
    def __init__(
            self,
            batch_size=1,
            cuda=0,
            model_type=None,
            temp_dir=None,
            **kwargs
    ):
        # Modified from original radgraph==0.1.13
        nn.Module.__init__(self)

        if cuda is None:
            cuda = -1
        if cuda >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cuda}")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        # End of modification

        self.cuda = cuda
        self.batch_size = batch_size

        if model_type is None:
            print("model_type not provided, defaulting to 'radgraph'")
            model_type = "radgraph"

        self.model_type = model_type.lower()

        assert model_type in ["radgraph", "radgraph-xl", "echograph"]

        if temp_dir is None:
            temp_dir = CACHE_DIR

        model_dir = os.path.join(temp_dir, model_type)

        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            try:
                archive_path = download_model(
                    repo_id="StanfordAIMI/RRG_scorers",
                    cache_dir=temp_dir,
                    filename=MODEL_MAPPING[model_type],
                )
            except Exception as e:
                raise Exception(e)

            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=model_dir)

        # Read config.
        config_path = os.path.join(model_dir, "config.json")
        config = json.load(open(config_path))
        config = DotMap(config)

        # Vocab
        vocab_dir = os.path.join(model_dir, "vocabulary")
        vocab_params = config.get("vocabulary", Params({}))
        vocab = Vocabulary.from_files(
            vocab_dir, vocab_params.get("padding_token"), vocab_params.get("oov_token")
        )

        # Tokenizer
        tok_indexers = {
            "bert": token_indexers.PretrainedTransformerMismatchedIndexer(
                model_name=config.dataset_reader.token_indexers.bert.model_name,
                max_length=config.dataset_reader.token_indexers.bert.max_length,
            )
        }
        self.reader = DyGIEReader(max_span_width=config.dataset_reader.max_span_width, token_indexers=tok_indexers)

        # Create embedder
        token_embedder = token_embedders.PretrainedTransformerMismatchedEmbedder(
            model_name=config.model.embedder.token_embedders.bert.model_name,
            max_length=config.model.embedder.token_embedders.bert.max_length
        )
        embedder = text_field_embedders.BasicTextFieldEmbedder({"bert": token_embedder})

        # Model
        model_dict = config.model
        for name in ["type", "embedder", "initializer", "module_initializer"]:
            del model_dict[name]

        model = dygie.DyGIE(vocab=vocab,
                            embedder=embedder,
                            **model_dict
                            )
        model_state_path = os.path.join(model_dir, "weights.th")
        model_state = torch.load(model_state_path, map_location=self.device, weights_only=True)
        model.load_state_dict(model_state, strict=True)
        model.eval()
        
        self.model = model.to(self.device)

class F1RadGraph(F1RadGraphCpu):
    def __init__(
            self,
            reward_level,
            model_type=None,
            cuda=0,
            **kwargs
    ):
        nn.Module.__init__(self)
        assert reward_level in ["simple", "partial", "complete", "all"]
        self.reward_level = reward_level
        self.radgraph = RadGraph(model_type=model_type, cuda=cuda, **kwargs)