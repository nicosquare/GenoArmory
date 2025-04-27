# -*- coding: utf-8 -*-
# @Time    : 2020/6/10
# @Author  : Linyang Li
# @Email   : linyangli19@fudan.edu.cn
# @File    : attack.py


import warnings
import os
import csv
import random
import transformers
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, Dataset
from transformers import BertConfig, AutoConfig, EsmConfig, AutoTokenizer
from transformers import AutoModelForSequenceClassification, EsmForSequenceClassification, AutoModelForMaskedLM
import copy
import argparse
import numpy as np

from quantization.range_estimators import OptMethod, RangeEstimators
from transformers_language.args import parse_args
from transformers_language.models.bert_attention import (
    AttentionGateType,
    EsmSelfAttentionwithExtra,
    Q4bitBertUnpadSelfAttentionWithExtras,

)
from transformers_language.models.quantized_dnabert import QuantizedBertForSequenceClassification
from transformers_language.models.quantized_nt import QuantizedEsmForSequenceClassification

from transformers_language.models.softmax import SOFTMAX_MAPPING
from transformers_language.quant_configs import get_quant_config
from transformers_language.utils import (
    count_params,
    kurtosis,
    pass_data_for_range_estimation,
    val_qparams,
)
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

filter_words = []
filter_words = set(filter_words)





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








def divide_dna_sequence(sequence, min_len=1, max_len=50):
    words = []
    start = 0
    
    # Continue splitting until the entire sequence is processed
    while start < len(sequence):
        # Randomly choose a word length between min_len and max_len
        word_length = random.randint(min_len, min_len + (max_len - min_len))
        
        # Ensure we do not exceed the sequence length
        word = sequence[start:start + word_length]
        words.append(word)
        
        # Move the starting point forward
        start += word_length
    
    # Join the words with spaces and return the result
    return ' '.join(words)

def get_tokenized_dna(sequence, tokenizer):
    tokens = tokenizer.tokenize(sequence)
    return ' '.join(tokens)

def get_sim_embed(embed_path, sim_path):
    id2word = {}
    word2id = {}

    with open(embed_path, 'r', encoding='utf-8') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in id2word:
                id2word[len(id2word)] = word
                word2id[word] = len(id2word) - 1

    cos_sim = np.load(sim_path)
    return cos_sim, word2id, id2word


def get_data_cls(data_path):
    lines = open(data_path, 'r', encoding='utf-8').readlines()[1:]
    features = []
    for i, line in enumerate(lines):
        split = line.strip('\n').split(',')
        label = int(split[-1])
        seq = split[0]

        features.append([seq, label])
    return features


class Feature(object):
    def __init__(self, seq_a, label):
        self.label = label
        self.seq = seq_a
        self.seq_down = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []


def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '')
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys


def _get_masked(words):
    len_text = len(words)
    masked_words = []
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    # list of words
    return masked_words


def get_important_scores(words, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length):
    masked_words = _get_masked(words)
    texts = [' '.join(words) for words in masked_words]  # list of text of masked words
    all_input_ids = []
    all_masks = []
    all_segs = []

    for text in texts:
        inputs = tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=max_length, )
        input_ids = inputs["input_ids"]
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + (padding_length * [0])
        #token_type_ids = token_type_ids + (padding_length * [0])
        attention_mask = attention_mask + (padding_length * [0])
        all_input_ids.append(input_ids)
        all_masks.append(attention_mask)
        #all_segs.append(token_type_ids)
    seqs = torch.tensor(all_input_ids, dtype=torch.long)
    masks = torch.tensor(all_masks, dtype=torch.long)
    segs = torch.tensor(all_segs, dtype=torch.long)
    seqs = seqs.to('cuda')

    eval_data = TensorDataset(seqs)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    leave_1_probs = []
    for batch in eval_dataloader:
        masked_input, = batch
        bs = masked_input.size(0)

        leave_1_prob_batch = tgt_model(masked_input)[0]  # B num-label
        leave_1_probs.append(leave_1_prob_batch)
    
    leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
    leave_1_probs = torch.softmax(leave_1_probs, -1)  #
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob
                     - leave_1_probs[:, orig_label]
                     +
                     (leave_1_probs_argmax != orig_label).float()
                     * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()
    return import_scores


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k
    
    if sub_len == 0:
        return words
        
    elif sub_len == 1:
        for (i,j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k

    substitutes = substitutes[0:12, 0:4] # maximum BPE candidates

    # find all possible candidates 
    #print(substitutes)
    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0] # N L vocab-size
    ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    
    for word in word_list:
        tokens = [tokenizer.convert_ids_to_tokens(int(i)) for i in word]
        #text = tokenizer.convert_tokens_to_string(tokens)
        text = ''.join(tokens)
        final_words.append(text)
    return final_words


def attack(feature, tgt_model, mlm_model, tokenizer, tokenizer_mlm, k, batch_size, max_length=512, cos_mat=None, w2i={}, i2w={}, use_bpe=1, threshold_pred_score=0.3):
    # MLM-process
    words, sub_words, keys = _tokenize(feature.seq, tokenizer)
    #words = [i for i in words[0]]
    # original label
    inputs = tokenizer.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length, )
    input_ids = torch.tensor(inputs["input_ids"])
    
    attention_mask = torch.tensor([1] * len(input_ids))
    seq_len = input_ids.size(0)
    orig_probs = tgt_model(input_ids.unsqueeze(0).to('cuda'),
                           attention_mask.unsqueeze(0).to('cuda'),
                           #token_type_ids.unsqueeze(0).to('cuda')
                           )[0].squeeze()
    orig_probs = torch.softmax(orig_probs, -1)
    orig_label = torch.argmax(orig_probs)
    current_prob = orig_probs.max()

    if orig_label != feature.label:
        feature.success = 3
        return feature
        

    sub_words = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]']
    input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
    word_predictions = mlm_model(input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)  # seq-len k

    word_predictions = word_predictions[1:len(sub_words) + 1, :]
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
    important_scores = get_important_scores(words, tgt_model, current_prob, orig_label, orig_probs,
                                            tokenizer, batch_size, max_length)

    feature.query += int(len(words))
    list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
    #print(list_of_index)
    final_words = copy.deepcopy(words)

    for top_index in list_of_index:
        if feature.change > int(0.4 * (len(words))):
            feature.success = 1  # exceed
            return feature
        if top_index[0] >= len(words):
            continue
        tgt_word = words[top_index[0]]
        if tgt_word in filter_words:
            continue

        # print(top_index)
        if top_index[0] >= len(keys) or keys[top_index[0]]==None:
            continue
        if keys[top_index[0]][0] > max_length - 2:
            continue


        substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k

        word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]
        
        substitutes = get_substitues(substitutes, tokenizer, mlm_model, use_bpe, word_pred_scores, threshold_pred_score)

        most_gap = 0.0
        candidate = None

        for substitute_ in substitutes:
            substitute = substitute_

            if substitute == tgt_word:
                continue  # filter out original word
            if '##' in substitute:
                continue  # filter out sub-word

            if substitute in filter_words:
                continue
            if substitute in w2i and tgt_word in w2i:
                if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                    continue
            temp_replace = final_words
            temp_replace[top_index[0]] = substitute
            
            #temp_text = tokenizer.convert_tokens_to_string(temp_replace)
            temp_text = ' '.join(temp_replace)
            inputs = tokenizer.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_length, )
            input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to('cuda')
            seq_len = input_ids.size(1)
            temp_prob = tgt_model(input_ids)[0].squeeze()
            feature.query += 1
            temp_prob = torch.softmax(temp_prob, -1)
            temp_label = torch.argmax(temp_prob)

            if temp_label != orig_label:
                feature.change += 1
                final_words[top_index[0]] = substitute
                feature.changes.append([keys[top_index[0]][0], substitute, tgt_word])
                feature.final_adverse = temp_text
                feature.success = 4
                return feature
            else:

                label_prob = temp_prob[orig_label]
                gap = current_prob - label_prob
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

        if most_gap > 0:
            feature.change += 1
            feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
            current_prob = current_prob - most_gap
            final_words[top_index[0]] = candidate

    feature.final_adverse = (''.join(final_words))
    
    feature.success = 2
    return feature


def evaluate(features):
    do_use = 0
    use = None
    sim_thres = 0
    # evaluate with USE

    if do_use == 1:
        cache_path = ''
        import tensorflow as tf
        import tensorflow_hub as hub
    
        class USE(object):
            def __init__(self, cache_path):
                super(USE, self).__init__()

                self.embed = hub.Module(cache_path)
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session()
                self.build_graph()
                self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

            def build_graph(self):
                self.sts_input1 = tf.placeholder(tf.string, shape=(None))
                self.sts_input2 = tf.placeholder(tf.string, shape=(None))

                sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
                sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
                self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
                clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
                self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

            def semantic_sim(self, sents1, sents2):
                sents1 = [s.lower() for s in sents1]
                sents2 = [s.lower() for s in sents2]
                scores = self.sess.run(
                    [self.sim_scores],
                    feed_dict={
                        self.sts_input1: sents1,
                        self.sts_input2: sents2,
                    })
                return scores[0]

            use = USE(cache_path)


    acc = 0
    origin_success = 0
    success4 = 0
    total = 0
    total_q = 0
    total_change = 0
    total_word = 0
    for feat in features:
        if feat.success > 2:

            if do_use == 1:
                sim = float(use.semantic_sim([feat.seq], [feat.final_adverse]))
                if sim < sim_thres:
                    continue
            
            acc += 1
            total_q += feat.query
            total_change += feat.change
            total_word += len(feat.seq.split(' '))

            if feat.success == 4:
                success4 += 1

            if feat.success == 3:
                origin_success += 1

        total += 1

    suc = float(acc / total)

    query = float(total_q / acc)
    change_rate = float(total_change / total_word)

    success4rate = float(success4 / total)

    origin_acc = 1 - origin_success / total
    after_atk = 1 - suc

    print('origin_acc/aft-atk-acc {:.6f}/ {:.6f}, success4rate {:.4f}, query-num {:.4f}, change-rate {:.4f}'.format(origin_acc, after_atk, success4rate, query, change_rate))


def dump_features(features, output):
    outputs = []

    for feature in features:
        outputs.append({'label': feature.label,
                        'success': feature.success,
                        'change': feature.change,
                        'num_word': len(feature.seq.split(' ')),
                        'query': feature.query,
                        'changes': feature.changes,
                        'seq_a': feature.seq.replace(' ', ''),
                        'adv': feature.final_adverse.replace(' ', ''),
                        })
    output_json = output
    json.dump(outputs, open(output_json, 'w'), indent=2)

    print('finished dump')


def run_attack():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="./data/xxx")
    parser.add_argument("--mlm_path", type=str, help="xxx mlm")
    parser.add_argument("--tgt_path", type=str, help="xxx classifier")

    parser.add_argument("--output_dir", type=str, help="train file")
    parser.add_argument("--use_sim_mat", type=int, help='whether use cosine_similarity to filter out atonyms')
    parser.add_argument("--start", type=int, help="start step, for multi-thread process")
    parser.add_argument("--end", type=int, help="end step, for multi-thread process")
    parser.add_argument("--num_label", type=int, )
    parser.add_argument("--use_bpe", type=int, )
    parser.add_argument("--k", type=int, )
    parser.add_argument("--threshold_pred_score", type=float, )


    parser.add_argument(
        "--attn_softmax",
        type=str,
        default="vanilla",
        help="Softmax variation to use in attention module.",
        choices=SOFTMAX_MAPPING.keys(),
    )
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--est_num_batches", type=int, default=16)
    parser.add_argument("--n_bits", type=int, default=8)
    parser.add_argument("--n_bits_act", type=int, default=8)
    parser.add_argument("--no_weight_quant", action="store_true")
    parser.add_argument("--no_act_quant", action="store_true")
    parser.add_argument("--qmethod_acts", type=str, default="asymmetric_uniform")
    parser.add_argument("--ranges_weights", type=str, default="minmax")
    parser.add_argument("--ranges_acts", type=str, default="running_minmax")
    parser.add_argument(
        "--percentile", type=float, default=None, help="Percentile (in %) for range estimation."
    )
    parser.add_argument("--quant_setup", type=str, default="all")

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=512,
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Path to a training file. If not provided, the training set will be split from the validation set.",
    )


    args = parser.parse_args()
    data_path = str(args.data_path)
    mlm_path = str(args.mlm_path)
    tgt_path = str(args.tgt_path)
    output_dir = str(args.output_dir)
    num_label = args.num_label
    use_bpe = args.use_bpe
    k = args.k
    start = args.start
    end = args.end
    threshold_pred_score = args.threshold_pred_score

    print('start process')

    tokenizer_mlm = AutoTokenizer.from_pretrained(
        mlm_path,
        model_max_length=128,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    try:
        tokenizer_tgt = AutoTokenizer.from_pretrained(tgt_path, trust_remote_code=True,)
    except:
        tokenizer_tgt = AutoTokenizer.from_pretrained(mlm_path, trust_remote_code=True,)

    config_atk = BertConfig.from_pretrained(mlm_path)
    
    mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_path, config=config_atk, trust_remote_code=True)
    mlm_model.to('cuda')

    config_tgt = AutoConfig.from_pretrained(tgt_path, num_labels=num_label, trust_remote_code=True,)
    tgt_model = AutoModelForSequenceClassification.from_pretrained(tgt_path, config=config_tgt, trust_remote_code=True,)
    

    for layer_idx in range(len(tgt_model.esm.encoder.layer)):
        old_self = tgt_model.esm.encoder.layer[layer_idx].attention.self
        print("----------------------------------------------------------")
        print("Inside BERT custom attention")
        print("----------------------------------------------------------")
        new_self = EsmSelfAttentionwithExtra(
            config_tgt,
            position_embedding_type=None,
            softmax_fn=SOFTMAX_MAPPING[args.attn_softmax],
        )

        # copy loaded weights
        if tgt_path is not None:
            new_self.load_state_dict(old_self.state_dict(), strict=False)
        tgt_model.esm.encoder.layer[layer_idx].attention.self = new_self

    embedding_size = tgt_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer_tgt) > embedding_size:
        print("Resizing token embeddings to fit tokenizer vocab size")
        tgt_model.resize_token_embeddings(len(tokenizer_tgt))

    
    
    # Quantize:
    if args.quantize:
        train_dataset = SupervisedDataset(tokenizer=tokenizer_tgt, 
                                          data_path=args.train_file, 
                                          kmer=-1)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer_tgt)
        
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
            num_workers=args.preprocessing_num_workers,
        )
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
            raise ValueError(f"Unknown weight range estimation: {args.ranges_weights}")

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
                click_config.act_quant.options = dict(opt_method=OptMethod.golden_section)

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
            raise NotImplementedError(f"Unknown act range estimation setting, '{args.ranges_acts}'")

        qparams = val_qparams(click_config)
        qparams["quant_dict"] = {}

        tgt_model = QuantizedEsmForSequenceClassification(tgt_model, **qparams)
        tgt_model.set_quant_state(
            weight_quant=click_config.quant.weight_quant, act_quant=click_config.quant.act_quant
        )

        # Range estimation
        pass_data_for_range_estimation(
            loader=train_dataloader,
            model=tgt_model,
            act_quant=click_config.quant.act_quant,
            max_num_batches=click_config.act_quant.num_batches,
        )
        tgt_model.fix_ranges()
        tgt_model.set_quant_state(
            weight_quant=click_config.quant.weight_quant, act_quant=click_config.quant.act_quant
        )



    tgt_model.to('cuda')
    features = get_data_cls(data_path)
    print('loading sim-embed')
    
    if args.use_sim_mat == 1:
        cos_mat, w2i, i2w = get_sim_embed('data_defense/counter-fitted-vectors.txt', 'data_defense/cos_sim_counter_fitting.npy')
    else:        
        cos_mat, w2i, i2w = None, {}, {}

    print('finish get-sim-embed')
    features_output = []

    with torch.no_grad():
        for index, feature in enumerate(features[start:end]):
            seq_a, label = feature
            seq_a = get_tokenized_dna(seq_a, tokenizer_tgt)
            feat = Feature(seq_a, label)
            print('\r number {:d} '.format(index) + tgt_path, end='')
            # print(feat.seq[:100], feat.label)
            feat = attack(feat, tgt_model, mlm_model, tokenizer_tgt, tokenizer_mlm, k, batch_size=32, max_length=128,
                          cos_mat=cos_mat, w2i=w2i, i2w=i2w, use_bpe=use_bpe,threshold_pred_score=threshold_pred_score)

            # print(feat.changes, feat.change, feat.query, feat.success)
            if feat.success > 2:
                print('success', end='')
            else:
                print('failed', end='')
            features_output.append(feat)

    print(" ")
    print("-----------")
    print(data_path)
    evaluate(features_output)

    dump_features(features_output, output_dir)


if __name__ == '__main__':
    run_attack()
