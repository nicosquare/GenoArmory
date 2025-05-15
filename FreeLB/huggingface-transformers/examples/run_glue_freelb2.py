# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

from comet_ml import Experiment
import sys
import argparse
import glob
import json
import logging
import os
import random
import sklearn
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.optim import AdamW

from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertConfig,
    WEIGHTS_NAME,
    get_linear_schedule_with_warmup,
)

from DNABERT2.bert_layers import DNABertForSequenceClassification
from DNABERT2.modeling_esm import EsmForSequenceClassification
from DNABERT2.modeling_esm2 import (
    EsmForSequenceClassification as EsmForSequenceClassification2,
)
from DNABERT2.modeling_hyena import HyenaDNAForSequenceClassification
from DNABERT2.modeling_mistral import MistralForSequenceClassification


from transformers import glue_processors as processors
import pdb
import sys
import torch.cuda.amp as amp


"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = (
        labels != -100
    )  # Exclude padding tokens (assuming -100 is the padding token ID)
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


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "dnabert": (BertConfig, DNABertForSequenceClassification, AutoTokenizer),
    "hyena": (AutoConfig, HyenaDNAForSequenceClassification, AutoTokenizer),
    "nt1": (AutoConfig, EsmForSequenceClassification, AutoTokenizer),
    "nt2": (AutoConfig, EsmForSequenceClassification2, AutoTokenizer),
    "og": (AutoConfig, MistralForSequenceClassification, AutoTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, experiment=None):
    """Train the model"""
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    if args.fp16:
        scaler = amp.GradScaler()
        # try:
        #     from apex import amp
        # except ImportError:
        #     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        # model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (
            len(train_dataloader) // args.gradient_accumulation_steps
        )
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps
        )

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoch",
            steps_trained_in_current_epoch,
        )

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    global_max_seq_len = -1
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            # using adaptive sequence length
            if args.model_type != "hyena":
                max_seq_len = torch.max(torch.sum(batch[1], 1)).item()
                batch = [t[:, :max_seq_len] for t in batch[:3]] + [batch[3]]
                if max_seq_len > global_max_seq_len:
                    global_max_seq_len = max_seq_len

                inputs = {"labels": batch[3]}

                if args.model_type in ["bert", "xlnet", "albert"]:
                        inputs["token_type_ids"] = batch[2]

                if args.model_type in ["dnabert", "nt1", "nt2", "og"]:
                    inputs["attention_mask"] = batch[1]
            else:
                max_seq_len = batch[0].shape[1]

                batch = [t[:, :max_seq_len] for t in batch[:1]] + [batch[1]]
                if max_seq_len > global_max_seq_len:
                    global_max_seq_len = max_seq_len

                inputs = {"labels": batch[1]}

            # ============================ Code for adversarial training=============
            # initialize delta
            if args.model_type == "dnabert":
                if isinstance(model, torch.nn.DataParallel):
                    embeds_init = model.module.bert.get_input_embeddings()(batch[0])
                else:
                    embeds_init = model.bert.get_input_embeddings()(batch[0])
            elif args.model_type == "nt1":
                if isinstance(model, torch.nn.DataParallel):
                    embeds_init = model.module.esm.get_input_embeddings()(batch[0])
                else:
                    embeds_init = model.esm.get_input_embeddings()(batch[0])
            elif args.model_type == "nt2":
                if isinstance(model, torch.nn.DataParallel):
                    embeds_init = model.module.esm.get_input_embeddings()(batch[0])
                else:
                    embeds_init = model.esm.get_input_embeddings()(batch[0])
            elif args.model_type == "og":
                if isinstance(model, torch.nn.DataParallel):
                    embeds_init = model.module.get_input_embeddings()(batch[0])
                else:
                    embeds_init = model.get_input_embeddings()(batch[0])
            elif args.model_type == "hyena":
                if isinstance(model, torch.nn.DataParallel):
                    embeds_init = model.module.get_input_embeddings()(batch[0])
                else:
                    embeds_init = model.get_input_embeddings()(batch[0])
            else:
                embeds_init = None
                print("Model type {} not specified.".format(args.model_type))
            if args.adv_init_mag > 0:
                if args.model_type in ["dnabert", "nt1", "nt2", "og"]:
                    input_mask = inputs["attention_mask"].to(embeds_init)
                    input_lengths = torch.sum(input_mask, 1)
                    # check the shape of the mask here..

                    if args.norm_type == "l2":
                        delta = torch.zeros_like(embeds_init).uniform_(
                            -1, 1
                        ) * input_mask.unsqueeze(2)
                        dims = input_lengths * embeds_init.size(-1)
                        mag = args.adv_init_mag / torch.sqrt(dims)
                        delta = (delta * mag.view(-1, 1, 1)).detach()
                    elif args.norm_type == "linf":
                        delta = torch.zeros_like(embeds_init).uniform_(
                            -args.adv_init_mag, args.adv_init_mag
                        ) * input_mask.unsqueeze(2)
                elif args.model_type == "hyena":
                    # For Hyena, we use sequence length directly since there's no attention mask
                    seq_length = embeds_init.size(1)
                    if args.norm_type == "l2":
                        delta = torch.zeros_like(embeds_init).uniform_(-1, 1)
                        dims = torch.tensor(seq_length * embeds_init.size(-1)).to(
                            "cuda"
                        )
                        mag = args.adv_init_mag / torch.sqrt(dims)
                        delta = (delta * mag.view(-1, 1, 1)).detach()
                    elif args.norm_type == "linf":
                        delta = torch.zeros_like(embeds_init).uniform_(
                            -args.adv_init_mag, args.adv_init_mag
                        )

            else:
                delta = torch.zeros_like(embeds_init)

            # the main loop
            dp_masks = None
            for astep in range(args.adv_steps):
                # (0) forward
                delta.requires_grad_()
                inputs["inputs_embeds"] = delta + embeds_init

                inputs["dp_masks"] = dp_masks

                
                # print(
                #     embeds_init.shape,
                #     file=sys.stderr,
                #     )
                # print(
                #     inputs["token_type_ids"].shape,
                #     file=sys.stderr,
                # )

                with torch.cuda.amp.autocast():
                    outputs, dp_masks = model(**inputs)
                    loss = outputs[
                        0
                    ]  # model outputs are always tuple in transformers (see doc)
                # (1) backward
                if args.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss = loss / args.adv_steps

                tr_loss += loss.item()

                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if astep == args.adv_steps - 1:
                    # further updates on delta
                    break

                # (2) get gradient on delta
                delta_grad = delta.grad.clone().detach()

                # (3) update and clip
                if args.norm_type == "l2":
                    denorm = torch.norm(
                        delta_grad.view(delta_grad.size(0), -1), dim=1
                    ).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta_norm = torch.norm(
                            delta.view(delta.size(0), -1).float(), p=2, dim=1
                        ).detach()
                        exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                        reweights = (
                            args.adv_max_norm / delta_norm * exceed_mask
                            + (1 - exceed_mask)
                        ).view(-1, 1, 1)
                        delta = (delta * reweights).detach()
                elif args.norm_type == "linf":
                    denorm = torch.norm(
                        delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")
                    ).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta = torch.clamp(
                            delta, -args.adv_max_norm, args.adv_max_norm
                        ).detach()
                else:
                    print("Norm type {} not specified.".format(args.norm_type))
                    exit()

                print(model)

                if args.model_type == "dnabert":
                    if isinstance(model, torch.nn.DataParallel):
                        embeds_init = model.module.bert.get_input_embeddings()(batch[0])
                    else:
                        embeds_init = model.bert.get_input_embeddings()(batch[0])
                elif args.model_type == "nt1":
                    if isinstance(model, torch.nn.DataParallel):
                        embeds_init = model.module.esm.get_input_embeddings()(batch[0])
                    else:
                        embeds_init = model.esm.get_input_embeddings()(batch[0])
                elif args.model_type == "nt2":
                    if isinstance(model, torch.nn.DataParallel):
                        embeds_init = model.module.esm.get_input_embeddings()(batch[0])
                    else:
                        embeds_init = model.esm.get_input_embeddings()(batch[0])
                elif args.model_type == "og":
                    if isinstance(model, torch.nn.DataParallel):
                        embeds_init = model.module.get_input_embeddings()(batch[0])
                    else:
                        embeds_init = model.get_input_embeddings()(batch[0])
                elif args.model_type == "hyena":
                    if isinstance(model, torch.nn.DataParallel):
                        embeds_init = model.module.get_input_embeddings()(batch[0])
                    else:
                        embeds_init = model.get_input_embeddings()(batch[0])

            # ============================ End (2) ==================

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping with mixed precision
                if args.fp16:
                    # Use scaler for mixed precision training (if fp16 is enabled)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                else:
                    # Normal precision (float32)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                # Perform optimizer step
                optimizer.step()

                # Update learning rate schedule
                scheduler.step()

                # Reset gradients for the next step
                model.zero_grad()

                # Increment the global step
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(
                            args,
                            model,
                            tokenizer,
                            global_step=global_step,
                            experiment=experiment,
                        )
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logs["max_seq_len"] = global_max_seq_len
                    if experiment is not None:
                        experiment.log_metric("TrainLoss", loss_scalar, global_step)
                    logging_loss = tr_loss

                    # for key, value in logs.items():
                    #     tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                    or global_step == args.max_steps - 1
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-last")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    )
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir
                    )

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", global_step=None, experiment=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (
        ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    )
    eval_outputs_dirs = (
        (args.output_dir, args.output_dir + "-MM")
        if args.task_name == "mnli"
        else (args.output_dir,)
    )

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            args, eval_task, tokenizer, evaluate=True
        )

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            if args.model_type != "hyena":
                # adaptive seq len
                max_seq_len = torch.max(torch.sum(batch[1], 1)).item()
                batch = [t[:, :max_seq_len] for t in batch[:3]] + [batch[3]]
            else:
                max_seq_len = batch[0].shape[1]
                batch = [t[:, :max_seq_len] for t in batch[:1]] + [batch[1]]

            with torch.no_grad():
                if args.model_type != "hyena":
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[3],
                    }
                else:
                    inputs = {
                        "input_ids": batch[0],
                        "labels": batch[1],
                    }

                if args.model_type in ["bert", "xlnet", "albert"]:
                    inputs["token_type_ids"] = batch[
                        2
                    ]  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps
        # if args.output_mode == "classification":
        #     preds = np.argmax(preds, axis=1)
        # elif args.output_mode == "regression":
        #     preds = np.squeeze(preds)
        result = compute_metrics((preds, out_label_ids))

        results = {"eval_loss": eval_loss, **result}

        if "best_criterion" not in results or result.get(
            "matthews_correlation", 0
        ) > results.get("best_criterion", 0):
            results["best_criterion"] = result.get("matthews_correlation", 0)

            output_dir = os.path.join(args.output_dir, "checkpoint-best")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        # with open(output_eval_file, "w") as writer:
        #     logger.info("***** Eval results {} *****".format(prefix))
        #     for key in sorted(result.keys()):
        #         logger.info("  %s = %s", key, str(result[key]))
        #         writer.write("%s = %s\n" % (key, str(result[key])))
        #         if experiment is not None:
        #             experiment.log_metric(key, result[key], global_step)

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    # Define label list manually
    label_list = list(range(0, args.num_label))

    # Construct cache filename
    cached_features_file = os.path.join(
        args.data_dir,
        f"cached_{'dev' if evaluate else 'train'}_{os.path.basename(args.model_name_or_path)}_{args.max_seq_length}_{task}",
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        # Manually load dataset
        file_path = os.path.join(args.data_dir, "dev.csv" if evaluate else "train.csv")
        examples = []
        lines = open(file_path, "r", encoding="utf-8").readlines()[1:]
        for i, line in enumerate(lines):
            split = line.strip("\n").split(",")
            label = int(split[-1])
            seq = split[0]

            examples.append([seq, label])

        # Convert examples to features
        features = []
        for text_a, label in examples:
            inputs = tokenizer(
                text_a,
                max_length=args.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
           
            if args.model_type == "hyena":
                features.append(
                    {
                        "input_ids": inputs["input_ids"].squeeze(0),
                        "label": label,
                    }
                )
            elif args.model_type == "dnabert":
                features.append(
                    {
                        "input_ids": inputs["input_ids"].squeeze(0),
                        "attention_mask": inputs["attention_mask"].squeeze(0),
                        "token_type_ids": inputs.get(
                            "token_type_ids",
                            torch.zeros_like(inputs["input_ids"]),
                        ),
                        "label": label,
                    }
                )
            else:
                features.append(
                    {
                        "input_ids": inputs["input_ids"].squeeze(0),
                        "attention_mask": inputs["attention_mask"].squeeze(0),
                        "token_type_ids": inputs.get(
                            "token_type_ids",
                            torch.zeros_like(inputs["input_ids"]),
                        ),
                        "label": label,
                    }
                )

        torch.save(features, cached_features_file)

    if args.model_type != "hyena":
        # Convert to Tensors and build dataset
        all_input_ids = torch.stack([f["input_ids"] for f in features])
        all_attention_mask = torch.stack([f["attention_mask"] for f in features])
        all_token_type_ids = torch.stack([f["token_type_ids"] for f in features])
        all_labels = torch.tensor(
            [label_list.index(f["label"]) for f in features], dtype=torch.long
        )
        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels
        )
    else:
        all_input_ids = torch.stack([f["input_ids"] for f in features])
        all_labels = torch.tensor(
            [label_list.index(f["label"]) for f in features], dtype=torch.long
        )

        dataset = TensorDataset(all_input_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: "
        + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
    parser.add_argument("--adv-lr", type=float, default=0)
    parser.add_argument("--adv-steps", type=int, default=1, help="should be at least 1")
    parser.add_argument("--adv-init-mag", type=float, default=0)
    parser.add_argument("--norm-type", type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument(
        "--adv-max-norm", type=float, default=0, help="set to 0 to be unlimited"
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--expname", type=str, default="default")
    parser.add_argument("--comet", default=False, action="store_true")
    parser.add_argument("--comet_key", default="", type=str)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0)
    parser.add_argument("--num_label", type=int)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.comet:
        experiment = Experiment(
            api_key=args.comet_key,
            project_name="pytorch-freelb",
            workspace="NLP",
            auto_param_logging=False,
            auto_metric_logging=False,
            parse_args=True,
            auto_output_logging=True,
        )
        experiment.disable_mp()  # Turn off monkey patching
        experiment.log_parameters(vars(args))
        experiment.set_name(args.expname)
    else:
        experiment = None

    assert args.adv_steps >= 1

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()

    args.output_mode = "classification"
    label_list = list(range(0, args.num_label))
    num_labels = args.num_label

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    print(args.model_name_or_path)
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        hidden_dropout_prob=args.hidden_dropout_prob,
        trust_remote_code=True,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        trust_remote_code=True,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        trust_remote_code=True,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, args.task_name, tokenizer, evaluate=False
        )
        global_step, tr_loss = train(
            args, train_dataset, model, tokenizer, experiment=experiment
        )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case
        )
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            )
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = (
                checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            )

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(
                args,
                model,
                tokenizer,
                prefix=prefix,
                global_step=global_step,
                experiment=experiment,
            )
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
