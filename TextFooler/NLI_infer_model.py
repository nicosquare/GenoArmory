import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, BertConfig, AutoConfig
from NLI_inf_data import NLIDataset_Hyena, NLIDataset_NT, NLIDataset_BERT, NLIDataset_OG


from quantization.range_estimators import OptMethod, RangeEstimators
from transformers_language.args import parse_args
from transformers_language.models.bert_attention import (
    AttentionGateType,
    BertUnpadSelfAttentionWithExtras,
    EsmSelfAttentionwithExtra,
    Q4bitBertUnpadSelfAttentionWithExtras,
)
from transformers_language.models.quantized_dnabert import (
    QuantizedBertForSequenceClassification,
)
from transformers_language.models.quantized_nt import (
    QuantizedEsmForSequenceClassification,
)
from transformers_language.models.softmax import SOFTMAX_MAPPING
from transformers_language.quant_configs import get_quant_config
from transformers_language.utils import (
    count_params,
    kurtosis,
    pass_data_for_range_estimation,
    val_qparams,
)
import transformers
from torch.utils.data import DataLoader, Dataset
import csv
import os
import json
from typing import List, Dict, Sequence
from dataclasses import dataclass



"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            json.dump(kmer, f)
        
    return kmer

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])



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
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

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

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )






class NLI_infer_Hyena(nn.Module):
    def __init__(self, pretrained_dir, nclasses, max_seq_length=128, batch_size=32, tokenizer_path=None):
        super(NLI_infer_Hyena, self).__init__()
        self.model_config = AutoConfig.from_pretrained(
            pretrained_dir, num_labels=nclasses, trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_dir, config=self.model_config, trust_remote_code=True
        ).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_Hyena(
            pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size, tokenizer_path=tokenizer_path
        )
        self.tokenizer = self.dataset.tokenizer

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        for input_ids, _ in dataloader:
            input_ids = input_ids.cuda()

            with torch.no_grad():
                # Get logits from the model
                logits = self.model(input_ids).logits
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class NLI_infer_NT(nn.Module):
    def __init__(self, pretrained_dir, nclasses, max_seq_length=128, batch_size=32, tokenizer_path=None, args=None):
        super(NLI_infer_NT, self).__init__()
        self.model_config = AutoConfig.from_pretrained(
            pretrained_dir, num_labels=nclasses
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_dir, config=self.model_config, trust_remote_code=True
        ).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_NT(
            pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size, tokenizer_path=tokenizer_path
        )
        self.tokenizer = self.dataset.tokenizer

        for layer_idx in range(len(self.model.esm.encoder.layer)):
            old_self = self.model.esm.encoder.layer[layer_idx].attention.self
            print("----------------------------------------------------------")
            print("Inside BERT custom attention")
            print("----------------------------------------------------------")
            new_self = EsmSelfAttentionwithExtra(
                self.model_config,
                position_embedding_type=None,
                softmax_fn=SOFTMAX_MAPPING[args.attn_softmax],
            )

            # copy loaded weights
            if pretrained_dir is not None:
                new_self.load_state_dict(old_self.state_dict(), strict=False)
            self.model.esm.encoder.layer[layer_idx].attention.self = new_self

        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            print("Resizing token embeddings to fit tokenizer vocab size")
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model = self.model.to('cuda')

        if args.quantize:
            click_config = get_quant_config()
            # override number of batches
            click_config.act_quant.num_batches = args.est_num_batches
            click_config.quant.n_bits = args.n_bits
            click_config.quant.n_bits_act = args.n_bits_act
            if args.no_weight_quant:
                click_config.quant.weight_quant = False
            if args.no_act_quant:
                click_config.quant.act_quant = False

            # Weight Ranges
            if args.ranges_weights == "minmax":
                pass
            elif args.ranges_weights in ("mse", "MSE"):
                click_config.quant.weight_quant_method = RangeEstimators.MSE
                click_config.quant.weight_opt_method = OptMethod.grid
            else:
                raise ValueError(
                    f"Unknown weight range estimation: {args.ranges_weights}"
                )

            # Acts ranges
            if args.percentile is not None:
                click_config.act_quant.options["percentile"] = args.percentile

            if args.ranges_acts == "running_minmax":
                click_config.act_quant.quant_method = RangeEstimators.running_minmax

            elif args.ranges_acts == "MSE":
                click_config.act_quant.quant_method = RangeEstimators.MSE
                if args.qmethod_acts == "symmetric_uniform":
                    click_config.act_quant.options = dict(opt_method=OptMethod.grid)
                elif args.qmethod_acts == "asymmetric_uniform":
                    click_config.act_quant.options = dict(
                        opt_method=OptMethod.golden_section
                    )

            elif args.ranges_acts.startswith("L"):
                click_config.act_quant.quant_method = RangeEstimators.Lp
                p_norm = float(args.ranges_acts.replace("L", ""))
                options = dict(p_norm=p_norm)
                if args.qmethod_acts == "symmetric_uniform":
                    options["opt_method"] = OptMethod.grid
                elif args.qmethod_acts == "asymmetric_uniform":
                    options["opt_method"] = OptMethod.golden_section
                click_config.act_quant.options = options

            else:
                raise NotImplementedError(
                    f"Unknown act range estimation setting, '{args.ranges_acts}'"
                )

            qparams = val_qparams(click_config)
            qparams["quant_dict"] = {}

            self.model = QuantizedEsmForSequenceClassification(self.model, **qparams)
            self.model.set_quant_state(
                weight_quant=click_config.quant.weight_quant,
                act_quant=click_config.quant.act_quant,
            )

            train_dataset = SupervisedDataset(tokenizer=self.tokenizer, 
                                      data_path=args.train_file, 
                                      kmer=-1)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
            
            train_dataloader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=data_collator,
                batch_size=args.batch_size,
                num_workers=args.preprocessing_num_workers,
            )
            # Range estimation
            pass_data_for_range_estimation(
                loader=train_dataloader,
                model=self.model,
                act_quant=click_config.quant.act_quant,
                max_num_batches=click_config.act_quant.num_batches,
            )
            self.model.fix_ranges()
            self.model.set_quant_state(
                weight_quant=click_config.quant.weight_quant,
                act_quant=click_config.quant.act_quant,
            )


    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, input_mask).logits
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class NLI_infer_BERT(nn.Module):
    def __init__(
        self, pretrained_dir, nclasses, max_seq_length=128, batch_size=32, args=None, tokenizer_path=None
    ):
        super(NLI_infer_BERT, self).__init__()
        self.model_config = BertConfig.from_pretrained(
            pretrained_dir, num_labels=nclasses
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_dir, config=self.model_config, trust_remote_code=True
        ).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(
            pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size, tokenizer_path=tokenizer_path
        )
        self.tokenizer = self.dataset.tokenizer

        for layer_idx in range(len(self.model.bert.encoder.layer)):
            old_self = self.model.bert.encoder.layer[layer_idx].attention.self
            print("----------------------------------------------------------")
            print("Inside BERT custom attention")
            print("----------------------------------------------------------")
            new_self = BertUnpadSelfAttentionWithExtras(
                self.model_config,
                position_embedding_type=None,
                softmax_fn=SOFTMAX_MAPPING[args.attn_softmax],
                ssm_eps=None,
                tau=None,
                max_seq_length=args.max_seq_length,
                skip_attn=False,
                fine_tuning=False,
            )

            # copy loaded weights
            if pretrained_dir is not None:
                new_self.load_state_dict(old_self.state_dict(), strict=False)
            self.model.bert.encoder.layer[layer_idx].attention.self = new_self

        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            print("Resizing token embeddings to fit tokenizer vocab size")
            self.model.resize_token_embeddings(len(self.tokenizer))

        if args.quantize:
            click_config = get_quant_config()
            # override number of batches
            click_config.act_quant.num_batches = args.est_num_batches
            click_config.quant.n_bits = args.n_bits
            click_config.quant.n_bits_act = args.n_bits_act
            if args.no_weight_quant:
                click_config.quant.weight_quant = False
            if args.no_act_quant:
                click_config.quant.act_quant = False

            # Weight Ranges
            if args.ranges_weights == "minmax":
                pass
            elif args.ranges_weights in ("mse", "MSE"):
                click_config.quant.weight_quant_method = RangeEstimators.MSE
                click_config.quant.weight_opt_method = OptMethod.grid
            else:
                raise ValueError(
                    f"Unknown weight range estimation: {args.ranges_weights}"
                )

            # Acts ranges
            if args.percentile is not None:
                click_config.act_quant.options["percentile"] = args.percentile

            if args.ranges_acts == "running_minmax":
                click_config.act_quant.quant_method = RangeEstimators.running_minmax

            elif args.ranges_acts == "MSE":
                click_config.act_quant.quant_method = RangeEstimators.MSE
                if args.qmethod_acts == "symmetric_uniform":
                    click_config.act_quant.options = dict(opt_method=OptMethod.grid)
                elif args.qmethod_acts == "asymmetric_uniform":
                    click_config.act_quant.options = dict(
                        opt_method=OptMethod.golden_section
                    )

            elif args.ranges_acts.startswith("L"):
                click_config.act_quant.quant_method = RangeEstimators.Lp
                p_norm = float(args.ranges_acts.replace("L", ""))
                options = dict(p_norm=p_norm)
                if args.qmethod_acts == "symmetric_uniform":
                    options["opt_method"] = OptMethod.grid
                elif args.qmethod_acts == "asymmetric_uniform":
                    options["opt_method"] = OptMethod.golden_section
                click_config.act_quant.options = options

            else:
                raise NotImplementedError(
                    f"Unknown act range estimation setting, '{args.ranges_acts}'"
                )

            qparams = val_qparams(click_config)
            qparams["quant_dict"] = {}

            self.model = QuantizedBertForSequenceClassification(self.model, **qparams)
            self.model.set_quant_state(
                weight_quant=click_config.quant.weight_quant,
                act_quant=click_config.quant.act_quant,
            )

            train_dataset = SupervisedDataset(tokenizer=self.tokenizer, 
                                      data_path=args.train_file, 
                                      kmer=-1)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
            
            train_dataloader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=data_collator,
                batch_size=args.batch_size,
                num_workers=args.preprocessing_num_workers,
            )
            # Range estimation
            pass_data_for_range_estimation(
                loader=train_dataloader,
                model=self.model,
                act_quant=click_config.quant.act_quant,
                max_num_batches=click_config.act_quant.num_batches,
            )
            self.model.fix_ranges()
            self.model.set_quant_state(
                weight_quant=click_config.quant.weight_quant,
                act_quant=click_config.quant.act_quant,
            )

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, input_mask, segment_ids).logits
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class NLI_infer_OG(nn.Module):
    def __init__(self, pretrained_dir, nclasses, max_seq_length=128, batch_size=32, tokenizer_path=None):
        super(NLI_infer_OG, self).__init__()
        self.model_config = AutoConfig.from_pretrained(
            pretrained_dir, num_labels=nclasses
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_dir, config=self.model_config, trust_remote_code=True
        ).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_OG(
            pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size, tokenizer_path=tokenizer_path
        )
        self.tokenizer = self.dataset.tokenizer

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, input_mask).logits
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)
