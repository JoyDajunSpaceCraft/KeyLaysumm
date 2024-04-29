import os
from dataclasses import dataclass, field
from typing import Optional
import argparse

# This is the simplification data from  https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker
# BAAI/bge-reranker-v2-m3 is the basic model

## Arguments
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="BAAI/bge-reranker-base", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # output_dir: str = field(
    #     default="bge_save", metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    # )


@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to corpus"}
    )
    train_group_size: int = field(default=8)
    max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for input text. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")



## Data
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding

# from .arguments import DataArguments


class TrainDatasetForCE(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer,
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        item = self.tokenizer.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        query = self.dataset[item]['query']
        pos = random.choice(self.dataset[item]['pos'])
        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)

        batch_data = []
        batch_data.append(self.create_one_example(query, pos))
        for neg in negs:
            batch_data.append(self.create_one_example(query, neg))

        return batch_data



@dataclass
class GroupCollator(DataCollatorWithPadding):
    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)



## Model
import logging

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

# from .arguments import ModelArguments, DataArguments

logger = logging.getLogger(__name__)


class CrossEncoder(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.config = self.hf_model.config
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.hf_model.gradient_checkpointing_enable(**kwargs)

    def forward(self, batch):
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits

        if self.training:
            scores = logits.view(
                self.train_args.per_device_train_batch_size,
                self.data_args.train_group_size
            )
            loss = self.cross_entropy(scores, self.target_label)

            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        state_dict = self.hf_model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.hf_model.save_pretrained(output_dir, state_dict=state_dict)


## Trainner
import logging
import os
from typing import Optional

import torch
from transformers.trainer import Trainer


logger = logging.getLogger(__name__)


class CETrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(f'MODEL {self.model.__class__.__name__} ' f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model: CrossEncoder, inputs):
        return model(inputs)['loss']


## Running 
import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, TrainingArguments
from transformers import (
    HfArgumentParser,
    set_seed,
)



logger = logging.getLogger(__name__)


# def main():
def main(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    # parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    num_labels = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    _model_class = CrossEncoder

    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = TrainDatasetForCE(data_args, tokenizer=tokenizer)
    _trainer_class = CETrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GroupCollator(tokenizer),
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer.train()
    trainer.save_model()

from dataclasses import dataclass, field
from transformers import TrainingArguments
from transformers import HfArgumentParser

# define models
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="BAAI/bge-reranker-base")
    config_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None

@dataclass
class DataArguments:
    train_data: str = field(default=None)
    train_group_size: int = 8
    max_len: int = 512

# Build up cases
model_args = ModelArguments(
    model_name_or_path="BAAI/bge-reranker-v2-m3",
    config_name=None,
    tokenizer_name=None,
    # cache_dir="./model_cache"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="elife")
    args = parser.parse_args()  
    train_data_dir = {
        "elife":"data/bge_train_wiki_plos.jsonl",
        "plos":"data/bge_train_wiki_elife.jsonl"
    }

    data_args = DataArguments(
        train_data=train_data_dir[args.data_type],
        train_group_size=10,
        max_len=128
    )

    training_args = TrainingArguments(
        output_dir=f"bge_models_{args.data_type}",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        fp16=True,
        learning_rate=5e-5,
        gradient_accumulation_steps=2,
        dataloader_drop_last=True,
        logging_steps=10,
        do_train=True,
        # overwrite_output_dir=False,
        weight_decay=0.01
    )
    # Training 
    main(model_args, data_args, training_args)



    # Evaluating 
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('bge_models')
    model = AutoModelForSequenceClassification.from_pretrained('bge_models')
    model.eval()

    pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        print(scores)


    # if __name__ == "__main__":
    #     main()