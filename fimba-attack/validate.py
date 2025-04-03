#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
import json
import logging
import math
import os
import random
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from transformers.models.bert.configuration_bert import BertConfig
import datasets
import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import DatasetDict, load_dataset, load_from_disk
from timm.utils import AverageMeter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)

from bert_layers import BertForSequenceClassification

from args import parse_args


from typing import Optional, Dict, Sequence, Tuple, List
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import csv
import sklearn

logger = logging.getLogger("validate_sc")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


EXTRA_METRICS = True


def attach_act_hooks(model):
    act_dict = OrderedDict()

    def _make_hook(name):
        def _hook(mod, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                inp = inp[0]
            act_dict[name] = (inp, out)

        return _hook

    for name, module in model.named_modules():
        module.register_forward_hook(_make_hook(name))
    return act_dict

"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        
        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    model: transformers.PreTrainedModel


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        labels, input_ids = tuple([instance[key] for instance in instances] for key in ( "labels", 'input_ids'))

        print(self)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        
        input_embeds = self.model.get_input_embeddings()(input_ids.to(self.model.device)).detach()
        labels = torch.Tensor(labels).long()
        return dict(
            input_embeds=input_embeds,
            labels=labels,
        )

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


def main():
    args = parse_args()
    logger.info(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in
    # this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up
    # all supported trackers in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir
        
        
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs
    )
    accelerator.init_trackers("tb_logs_validation", init_kwargs={"wandb":{"name":args.run_name}})

    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Prepare HuggingFace config
    # In distributed training, the .from_pretrained methods guarantee that only one local process
    # can concurrently download model & vocab.
    config_kwargs = {
        "cache_dir": args.model_cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # Display config after changes
    logger.info("HuggingFace config after user changes:")
    logger.info(str(config))

    # Load tokenizer
    tokenizer_kwargs = {
        "trust_remote_code": args.trust_remote_code,
    }
    if args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this "
            "script. You can do it from another script, save it, and load it from here, "
            "using --tokenizer_name."
        )

    # Load and prepare model
    if args.model_name_or_path:
        config = BertConfig.from_pretrained(args.model_name_or_path)
        model = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.model_cache_dir,
            trust_remote_code=args.trust_remote_code
        )
    else:
        logger.info("Training new model from scratch")
        model = BertForSequenceClassification.from_config(config)


    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print("Resizing token embeddings to fit tokenizer vocab size")
        model.resize_token_embeddings(len(tokenizer))

    
    

    
    # Check sequence length
    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` "
                f"({tokenizer.model_max_length}). Picking 1024 instead. You can change that "
                f"default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum "
                f"length for the model ({tokenizer.model_max_length}). Using "
                f"max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=args.train_file, 
                                      kmer=-1)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=args.validation_file, 
                                     kmer=-1)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, model=model)
    



    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.preprocessing_num_workers,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=args.preprocessing_num_workers,
        shuffle=True,
    )    

    # Prepare everything with our `accelerator`.
    model,train_dataloader, eval_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader)

    logger.info("FP model:")
    logger.info(model)



    # attach hooks for activation stats (if needed)
    act_dict = {}
    if EXTRA_METRICS:
        act_dict = attach_act_hooks(model)

    num_layers = len(model.bert.encoder.layer)
    act_inf_norms = OrderedDict()
    act_kurtoses = OrderedDict()

    # *** Evaluation ***
    model.eval()
    losses = []
    all_logits = []
    all_labels = []

    for batch_idx, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        logits = outputs.logits  # Extract logits
        labels = batch["labels"]  # Assuming labels are in the batch

        # Gather loss for metrics
        loss_ = accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size))
        losses.append(loss_.detach().cpu())  # Detach and move to CPU to free GPU memory

        # Gather logits and labels for custom metric computation
        gathered_logits = accelerator.gather_for_metrics(logits).detach().cpu()
        gathered_labels = accelerator.gather_for_metrics(labels).detach().cpu()

        all_logits.append(gathered_logits)
        all_labels.append(gathered_labels)

        
    # Concatenate all gathered tensors
    losses = torch.cat(losses)
    logits = torch.cat(all_logits).numpy()  # Convert to numpy
    labels = torch.cat(all_labels).numpy()  # Convert to numpy
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    
    # Calculate custom metrics
    metrics = compute_metrics((logits, labels))

    # Logging
    logger.info(f"perplexity: {perplexity:.4f}")
    logger.info(f"loss: {eval_loss:.4f}")
    logger.info(f"accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"f1: {metrics['f1']:.4f}")
    logger.info(f"matthews_correlation: {metrics['matthews_correlation']:.4f}")
    logger.info(f"precision: {metrics['precision']:.4f}")
    logger.info(f"recall: {metrics['recall']:.4f}")


    

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    main()
