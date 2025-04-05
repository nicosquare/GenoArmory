from transformers import DataProcessor, InputExample, InputFeatures
import os
from transformers import logging

logger = logging.get_logger(__name__)

import os
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union
import csv
import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from transformers import PreTrainedTokenizerBase
from transformers import logging
from transformers import glue_convert_examples_to_features, glue_output_modes, glue_processors
from transformers import InputFeatures

logger = logging.get_logger(__name__)

classification_tasks_num_labels = {"imdb": 2, "imdb-adv":2, "mr": 2, 'mnli': 3, 'mnli-adv':3, 'mr-adv': 2, 'sst': 2, 'sst-adv':2, 'sst-pair':2,'mr-pair':2,'cr-pair':2,"cr": 2}


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if set_type == "test" else 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class Sst2ProcessorAdv(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5", "6", "7"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if set_type == "test" else 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2ProcessorPair(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if set_type == "test" else 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = None if set_type == "test" else line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        csv.field_size_limit(500 * 1024 * 1024)
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        csv.field_size_limit(500 * 1024 * 1024)
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        csv.field_size_limit(500 * 1024 * 1024)
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = None if set_type.startswith("test") else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MnliProcessorAdv(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        csv.field_size_limit(500 * 1024 * 1024)
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        csv.field_size_limit(500 * 1024 * 1024)
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        csv.field_size_limit(500 * 1024 * 1024)
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral",
                "contradiction_simplified", "entailment_simplified", "neutral_simplified",
                "contradiction_attacked", "entailment_attacked", "neutral_attacked",
                "contradiction_simplified_attacked", "entailment_simplified_attacked", "neutral_simplified_attacked"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[0]
            text_b = line[1]
            label = None if set_type.startswith("test") else line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")

class MnliMismatchedProcessorAdv(MnliProcessorAdv):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")


class IMDBProcessor(DataProcessor):
    """Processor for the IMDB data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        lines = lines[1:]
        text_index = 1 if test_mode else 0
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = None if test_mode else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class IMDBProcessorAdv(DataProcessor):
    """Processor for the IMDB data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5", "6", "7"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        lines = lines[1:]
        text_index = 1 if test_mode else 0
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = None if test_mode else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class MRProcessor(DataProcessor):
    """Processor for the MR data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        lines = lines[1:]
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class MRProcessorPair(MRProcessor):
    """Processor for the MR data set (GLUE version)."""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        lines = lines[1:]
        examples = []
        for (i, line) in enumerate(lines):
            if line[0] == 'sentence':
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



class MRProcessorAdv(DataProcessor):
    """Processor for the MR data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5", "6", "7"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        lines = lines[1:]
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


classification_processors = {'imdb': IMDBProcessor, 'imdb-adv':IMDBProcessorAdv, 'mr': MRProcessor, 'mnli': MnliProcessor,
                             "mnli-mm": MnliMismatchedProcessor,
                             "mnli-adv":MnliProcessorAdv, "mnli-mm-adv": MnliMismatchedProcessorAdv,
                             "mr-adv": MRProcessorAdv, 'sst': Sst2Processor, 'sst-adv': Sst2ProcessorAdv, 'sst-pair':Sst2ProcessorPair,'mr-pair':MRProcessorPair,
                             'cr':MRProcessor,'cr-pair':MRProcessorPair}


@dataclass
class ClassificationDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command
    line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    nlabels: int = field(
        default=2, metadata={"help": "Number of class number"}
    )


    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class ClassificationDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    args: ClassificationDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
            self,
            args: ClassificationDataTrainingArguments,
            model_args,
            tokenizer: PreTrainedTokenizerBase,
            limit_length: Optional[int] = None,
            mode: Union[str, Split] = Split.train,
            cache_dir: Optional[str] = None
    ):
        warnings.warn(
            "This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
            "library. You can have a look at this example script for pointers: "
            "https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py",
            FutureWarning,
        )
        self.args = args
        label_list = list(range(0,args.nlabels + 6))
    
    
        # Construct cache filename
        cached_features_file = os.path.join(
            args.data_dir,
            f"cached_{'dev' if {mode} else 'train'}_{os.path.basename(model_args.model_name_or_path)}_{args.max_seq_length}"
        )
        self.output_mode = "classification"
        
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            
            # Manually load dataset
            file_path = os.path.join(args.data_dir, "dev.tsv" if mode=='dev' else "train.tsv")
            examples = []
            lines = open(file_path, 'r', encoding='utf-8').readlines()[1:]
            for i, line in enumerate(lines):
                try:
                    split = line.strip('\n').split('\t')
                    label = int(split[-1])
                    seq = split[0]
                    examples.append([seq, label])
                except:
                    split = line.strip('\n').split(',')
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
                    return_tensors="pt"
                )
                
                if model_args.model_type == "hyena":
                    features.append({
                        "input_ids": inputs["input_ids"].squeeze(0),
                        "label": label
                    })
                else:
                    features.append({
                        "input_ids": inputs["input_ids"].squeeze(0),
                        "attention_mask": inputs["attention_mask"].squeeze(0),
                        "token_type_ids": inputs.get("token_type_ids", torch.zeros_like(inputs["input_ids"])),
                        "label": label
                    })
            torch.save(features, cached_features_file)
        
        if model_args.model_type == "hyena":
            all_input_ids = torch.stack([f["input_ids"] for f in features])
            all_labels = torch.tensor([label_list.index(f["label"]) for f in features], dtype=torch.long)
        elif model_args.model_type == "nt1" or model_args.model_type == "nt2" or model_args.model_type == "og":
            all_input_ids = torch.stack([f["input_ids"] for f in features])
            all_attention_mask = torch.stack([f["attention_mask"] for f in features])
            all_labels = torch.tensor([label_list.index(f["label"]) for f in features], dtype=torch.long)
        else:
            all_input_ids = torch.stack([f["input_ids"] for f in features])
            all_attention_mask = torch.stack([f["attention_mask"] for f in features])
            all_token_type_ids = torch.stack([f["token_type_ids"] for f in features])
            all_labels = torch.tensor([label_list.index(f["label"]) for f in features], dtype=torch.long)
        
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list
