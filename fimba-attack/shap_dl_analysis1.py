'''
This script is used to precalculate SHAP values for the models trained on the TCGA and COVID datasets.
In our attack we compute SHAP values in real time on the target sample, here to analyze the SHAP values in general we compute them for DL models in bulk.
It is highly recommended to run this script on a machine with a GPU, as it is very computationally expensive and use a sample of the dataset.
'''



import shap
import pandas as pd
import os
import pickle
import torch
import argparse
import gc
import sys
from tqdm import tqdm
from models import *
from transformers import AutoTokenizer, AutoConfig, BertConfig
import scipy as sp

from bert_layers import BertForSequenceClassification

from torch.nn import DataParallel

import torch
import random
import sklearn
import os

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

@contextlib.contextmanager
def change_dir(path):
    _oldcwd = os.getcwd()
    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(_oldcwd)


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


def load_and_cache_examples(args, task, tokenizer, model, evaluate=False):
    # Define label list manually
    label_list = list(range(0,args.num_label))
    
    
    # Construct cache filename
    cached_features_file = os.path.join(
        args.data_dir,
        f"cached_{'dev' if evaluate else 'train'}_{os.path.basename(args.model_name_or_path)}_{args.max_seq_length}_{task}"
    )
    
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        
        features = torch.load(cached_features_file)
    else:
        
        
        # Manually load dataset
        file_path = os.path.join(args.data_dir, "dev.csv" if evaluate else "train.csv")
        examples = []
        lines = open(file_path, 'r', encoding='utf-8').readlines()[1:]
        for i, line in enumerate(lines):
            split = line.strip('\n').split(',')
            label = int(split[-1])
            seq = split[0]

            examples.append([seq, label])
                
        # Convert examples to features
        features = []
        x_data = []
        for text_a, label in examples:
            inputs = tokenizer(
                text_a,
                max_length=args.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                features.append({
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0),
                    "token_type_ids": inputs.get("token_type_ids", torch.zeros_like(inputs["input_ids"])),
                    "input_embeds": model.model.get_input_embeddings()(inputs["input_ids"].to(model.model.device)).detach(),  # Add this line
                    "label": label
                })

        
        torch.save(features, cached_features_file)
    
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.stack([f["input_ids"] for f in features])
    all_attention_mask = torch.stack([f["attention_mask"] for f in features])
    all_token_type_ids = torch.stack([f["token_type_ids"] for f in features])
    all_labels = torch.tensor([label_list.index(f["label"]) for f in features], dtype=torch.long)
    all_input_embeds = torch.stack([f["input_embeds"] for f in features])
    
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_input_embeds, all_labels)

    return dataset, all_labels, features

def prepare_shap_inputs(features, device):
    input_embeds = torch.stack([f["input_embeds"] for f in features]).to(device)
    input_embeds = input_embeds.squeeze(1)
    return input_embeds
    



class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, input_embeds):  # Modified to directly take input_embeds for PyTorchDeep compatibility
        outputs = self.model(input_embeds=input_embeds)
        return outputs.logits



if __name__ == '__main__':

    print(torch.cuda.memory_summary())
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
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
        required=True
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
        "--batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    
    parser.add_argument('--num_label', type=int)
    parser.add_argument('--dataset_name', type=str, default='0')
    parser.add_argument('--model_name', type=str, default='DNABERT2')

    args = parser.parse_args()

    dataset = args.data_dir # covid
    model_name = args.model_name_or_path # resnet, transformer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    config = BertConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_label,
        cache_dir=args.cache_dir if args.cache_dir else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        trust_remote_code=True,
    )
    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        trust_remote_code=True,
    )

    print(f'Loaded model: {model_name} for {dataset} dataset\n')


    wrapmodel = WrappedModel(model)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        wrapmodel = nn.DataParallel(wrapmodel)


    

    wrapmodel.to(device)
    wrapmodel.eval()
    
    train_dataset, ytrain, train_data = load_and_cache_examples(args, args.task_name, tokenizer, wrapmodel, evaluate=False)

    test_dataset, ytest, tst_data = load_and_cache_examples(args, args.task_name, tokenizer, wrapmodel, evaluate=True)

    

    print(f'Dataset {train_dataset} of shape {ytrain.shape} loaded successfully!\n')
    print('Starting SHAP analysis...\n')
    print(torch.cuda.memory_summary())

    # Get the first 1000 samples from training data for background
    train_data_subset = train_data[:1000]
    test_data_subset = tst_data  # Also take 1000 test samples for consistency
    
    vis_dataset = pd.DataFrame({
        'input_embeds': train_data_subset,  # Or select relevant data
    })


    def f(x):
        # x is a list of text sequences
        # Tokenize the sequences
        inputs = tokenizer(x, padding="max_length", max_length=args.max_seq_length, 
                         truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()
        
        # Convert to probabilities and get logits for positive class
        scores = (np.exp(logits).T / np.exp(logits).sum(-1)).T
        val = sp.special.logit(scores[:, 1])  # use one vs rest logit units
        return val

    # build an explainer using a token masker
    explainer = shap.Explainer(f, tokenizer)

    # Convert input_ids to text sequences for background
    train_sequences = []
    for data in train_data_subset:
        sequence = tokenizer.decode(data["input_ids"], skip_special_tokens=True)
        train_sequences.append(sequence)

    # Initialize progress bar
    pbar = tqdm(total=len(test_data_subset))
    shap_values = []
    flag = True

    # Calculate SHAP values for each test sample
    for i in range(len(test_data_subset)):
        # Convert input_ids to text sequence
        sequence = tokenizer.decode(test_data_subset[i]["input_ids"], skip_special_tokens=True)
        x = [sequence]
        
        # Calculate SHAP values for the i-th sample
        shap_values_i = explainer.shap_values(x, check_additivity=False)
        
        if flag:
            print(f'Shap calculation check, val = {shap_values_i}')
            flag = False
        
        del x
        gc.collect()
        torch.cuda.empty_cache()
        
        shap_values.append(shap_values_i[0])
        pbar.set_description(f'Calculating SHAP values for samples {i+1} to {min(i+1, len(test_data_subset))}')
        pbar.update(1)
        pbar.refresh()

    pbar.close()
    
    # Concatenate all SHAP values
    shapvalues = np.concatenate(shap_values, axis=0)
    print(f'SHAP values shape: {shapvalues.shape}')
    #shapvalues = explainer.shap_values(test_data.unsqueeze(1).to(device).to(torch.float32))
    expected_value = explainer.expected_value[0]
    print(expected_value)
    

    if dataset=='tcga':
        base_values = np.full((2881),expected_value)
    else:
        base_values = np.full((2000),expected_value)
    print('SHAP analysis finished\n')

    shap_arr = shapvalues.reshape(shapvalues.shape[0],1,-1)
    shap_arr = shap_arr.transpose(0,2,1).squeeze()

    

    exp_model = shap.Explanation(shap_arr, 
                    base_values,
                    data=vis_dataset.values,
                    feature_names=vis_dataset.columns)
    print('Saving SHAP values...\n')
    with open('shap_dicts/shap_'+args.model_name+'_2_'+args.dataset_name+'.pkl', 'wb') as f:
        pickle.dump(shapvalues, f)


