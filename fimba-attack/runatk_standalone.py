import shap
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoConfig, BertConfig, MistralConfig, AutoModelForSequenceClassification
import argparse
import numpy as np
import pandas as pd
from bert_layers import BertForSequenceClassification
from modeling_esm2 import EsmForSequenceClassification as EsmForSequenceClassification2
import pickle
import torch
import random
import sklearn
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import BallTree
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.nn as nn
from scipy.spatial import KDTree
import json
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


def prepare_shap_inputs(features, device):
    input_embeds = torch.stack([f["input_embeds"] for f in features]).to(device)
    input_embeds = input_embeds.squeeze(1)
    return input_embeds
    

class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, input_embeds):  # Modified to directly take input_embeds for PyTorchDeep compatibility
        outputs = self.model(inputs_embeds=input_embeds)
        return outputs.logits

def df_dl_predict(args, model, testloader, device):
    preds, labels_list = [], []

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        if args.model_type == 'dnabert':
            for input_ids, attention_mask, token_type_ids, input_embeds, labels in testloader:
                labels = labels.to(device)

                token_type_ids = token_type_ids.squeeze(1)

                inputs = {
                    'input_ids': input_ids.to(device),
                    'attention_mask': attention_mask.to(device),
                    'token_type_ids': token_type_ids.to(device)
                }

                
                outputs = model(**inputs).logits
                preds.append(outputs.cpu().detach().numpy())
                labels_list.append(labels.cpu().detach().numpy())
        elif args.model_type == 'hyena':
            for input_ids, input_embeds, labels in testloader:
                labels = labels.to(device)

                inputs = {
                    'input_ids': input_ids.to(device),
                }

                outputs = model(**inputs).logits
                preds.append(outputs.cpu().detach().numpy())
                labels_list.append(labels.cpu().detach().numpy())
        else:
            for input_ids, attention_mask, input_embeds, labels in testloader:
                labels = labels.to(device)

                inputs = {
                    'input_ids': input_ids.to(device),
                    'attention_mask': attention_mask.to(device),
                }

                outputs = model(**inputs).logits
                preds.append(outputs.cpu().detach().numpy())
                labels_list.append(labels.cpu().detach().numpy())
    preds = np.vstack(preds)
    labels_list = np.hstack(labels_list)

    print(model)
    metrics = calculate_metric_with_sklearn(preds, labels_list)

    return preds, metrics

def df_dl_predict2(model, testloader, device):
    preds, labels_list = [], []

    model = model.to(device)
    model.eval()


    with torch.no_grad():
        for input_embeds, labels in testloader:
            labels = labels.to(device)


            inputs = {
                'inputs_embeds': input_embeds.to(device),
            }


            outputs = model(**inputs).logits
            preds.append(outputs.cpu().detach().numpy())
            labels_list.append(labels.cpu().detach().numpy())

    preds = np.vstack(preds)
    labels_list = np.hstack(labels_list)

    metrics = calculate_metric_with_sklearn(preds, labels_list)

    return preds, metrics


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
                if args.model_type == 'dnabert':
                    features.append({
                        "input_ids": inputs["input_ids"].squeeze(0),
                        "attention_mask": inputs["attention_mask"].squeeze(0),
                        "token_type_ids": inputs.get("token_type_ids", torch.zeros_like(inputs["input_ids"])),
                        "input_embeds": model.get_input_embeddings()(inputs["input_ids"].to(model.device)).detach(),  # Add this line
                        "label": label
                    })
                elif args.model_type == 'hyena':
                    features.append({
                        "input_ids": inputs["input_ids"].squeeze(0),
                        "input_embeds": model.get_input_embeddings()(inputs["input_ids"].to(model.device)).detach(),  # Add this line
                        "label": label
                    }) 
                else:
                    features.append({
                        "input_ids": inputs["input_ids"].squeeze(0),
                        "attention_mask": inputs["attention_mask"].squeeze(0),
                        "input_embeds": model.get_input_embeddings()(inputs["input_ids"].to(model.device)).detach(),  # Add this line
                        "label": label
                    })
        
        torch.save(features, cached_features_file)
    
    if args.model_type == 'hyena':
        all_input_ids = torch.stack([f["input_ids"] for f in features])
        all_input_embeds = torch.stack([f["input_embeds"] for f in features])
        all_labels = torch.tensor([label_list.index(f["label"]) for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_embeds, all_labels)
    elif args.model_type == 'dnabert':
        all_input_ids = torch.stack([f["input_ids"] for f in features])
        all_attention_mask = torch.stack([f["attention_mask"] for f in features])
        all_token_type_ids = torch.stack([f["token_type_ids"] for f in features])
        all_labels = torch.tensor([label_list.index(f["label"]) for f in features], dtype=torch.long)
        all_input_embeds = torch.stack([f["input_embeds"] for f in features])
        
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_input_embeds, all_labels)
    else:
        all_input_ids = torch.stack([f["input_ids"] for f in features])
        all_attention_mask = torch.stack([f["attention_mask"] for f in features])
        all_labels = torch.tensor([label_list.index(f["label"]) for f in features], dtype=torch.long)
        all_input_embeds = torch.stack([f["input_embeds"] for f in features])
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_input_embeds, all_labels)
    return dataset, all_labels, features

def shap_attack_mp(args): # this is a self contained function that can be run in parallel, shap values are computed in real time here for each sample
    #topf, shap_importance, model, neg_data_test, pos_data_test, xtest, ytest, tree_fps, tree_fns, fps, fns, increase_fn, model_type = args
    if args.model_type == 'og':
        config = MistralConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=args.num_label,
            cache_dir=args.cache_dir if args.cache_dir else None,
            trust_remote_code=True,
        )
    elif args.model_type == 'dnabert':
        config = BertConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=args.num_label,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
            trust_remote_code=True,
        )
    else:
        config = AutoConfig.from_pretrained(
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

    if args.model_type == 'nt1':
        config.token_dropout = False

    if args.model_type == 'dnabert':
        model = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
            trust_remote_code=True,
        )
    elif args.model_type == 'nt2':
        model = EsmForSequenceClassification2.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
            trust_remote_code=True,
        )
    

    wrapmodel = WrappedModel(model)

    wrapmodel.to('cuda')
    wrapmodel.eval()

    train_dataset, ytrain, train_data = load_and_cache_examples(args, args.task_name, tokenizer, model, evaluate=False)
    test_dataset, ytest, xtest = load_and_cache_examples(args, args.task_name, tokenizer, model, evaluate=True)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    train_data = prepare_shap_inputs(train_data[:args.batch_size], 'cuda')  # Small sample for background data
    xtest = prepare_shap_inputs(xtest, 'cuda')

    x_copy = xtest.clone()
    vis_dataset = pd.DataFrame({
        'input_embeds': xtest.cpu().numpy().tolist(),  # Or select relevant data
    })

    FP_count = []
    FN_count = []
    Accuracy = []
    
    precision = args.max_seq_length
    increase_fn = args.increase_fn
    print('Loaded args')
    
    preds, metrics1 = df_dl_predict(args, model, testloader, device='cuda')

    print(metrics1)
    

    explainer = shap.DeepExplainer(wrapmodel, train_data)
    
    
    with open(args.shap_file, 'rb') as file:
        exp_model = pickle.load(file)

    shap_values = exp_model.values  # SHAP values array
    shap_importance = np.abs(shap_values).mean(axis=0)


    shap_importance_df = pd.DataFrame(
        data=shap_importance.reshape(-1, 1),
        columns=exp_model.feature_names
    )


    top_feat = shap_importance_df.iloc[:args.topf]


    top_feat_names = top_feat.columns.values

    zero_idx_test = torch.where(ytest == 0)[0]
    one_idx_test = torch.where(ytest == 1)[0]

    pred_labels  = np.argmax(preds, axis=1)
    fp_indices = np.where((pred_labels == 1) & (ytest.numpy() == 0))[0]  # False Positives
    fn_indices = np.where((pred_labels == 0) & (ytest.numpy() == 1))[0]  # False Negatives

    fps_data = xtest[fp_indices]  # Data for False Positives
    fns_data = xtest[fn_indices]  # Data for False Negatives



    fps_data_pooled = torch.mean(fps_data, dim=1).cpu().numpy()
    fns_data_pooled = torch.mean(fns_data, dim=1).cpu().numpy()
    
    tree_fps = KDTree(fps_data_pooled)

   
    tree_fns = KDTree(fns_data_pooled)

    

    x_copy = xtest.clone()  # get a fresh set
    print('Running SHAP attack...')
    
    # Debug: Print original and modified embeddings for a few samples
    print("Original embeddings shape:", xtest.shape)
    print("Original embeddings mean:", torch.mean(xtest).item())
    print("Original embeddings std:", torch.std(xtest).item())

    expected_value = explainer.expected_value[0]
    base_values = np.full((shap_values.shape),expected_value)

    for index, row in tqdm(enumerate(x_copy), total=len(x_copy), desc="Processing"):
        row = row.unsqueeze(0)
        row = row.requires_grad_(True)
        shapvalues = explainer.shap_values(row, check_additivity=False)

        shap_arr = shapvalues
        
        # shap_arr = shapvalues.reshape(shapvalues.shape[0],1,-1)
        # shap_arr = shap_arr.transpose(0,2,1).squeeze()32

        
        exp_model2 = shap.Explanation(shap_arr,  
                base_values,
                data=vis_dataset.values,
                feature_names=vis_dataset.columns)     

        shap_values2 = exp_model2.values  # SHAP values array

        # print(shap_values2)
        shap_importance2 = np.abs(shap_values2).mean(axis=0)
        # print(shap_importance2)

        shap_importance_df2 = pd.DataFrame(
            data=shap_importance2.reshape(-1, 1),
            columns=exp_model2.feature_names
        )

            
        top_feat = shap_importance_df.iloc[:args.topf]
        top_feat_names = top_feat.columns.values
        
        
        
        if index in zero_idx_test:
            row_pooled = torch.mean(row, dim=1).detach().cpu().numpy()
    
            id_neg = tree_fps.query(row_pooled, k=[2])
            
            if id_neg[0][
                0] == 0:  # this means we are getting fps vector TODO: remove fps from the test set before running the algorithm
                continue
            else:
                vector_id = id_neg[1][0]
                a0 = row_pooled
                try:
                    if vector_id < len(fns_data_pooled) and len(fns_data_pooled) > 0:
                        a1 = fps_data_pooled[vector_id]
                    else:
                        continue  # Skip if vector_id is out of bounds

                        
                    a0_val = a0
                    a1_val = a1
                    # Increase the interpolation range to make more significant changes
                    sample_space = np.linspace(a0_val, a1_val, precision)
                    # Take the last value in the interpolation to make more significant changes
                    x_copy[index] = torch.from_numpy(sample_space[-1]).to(x_copy.device)
                    # print(f"Feature: {feature} | Original vector value: {a0_val} | Closest FP vector value {a1_val} | Sample space: {sample_space[-2]}")
                except:
                    continue

        if increase_fn:
            if index in one_idx_test:  # positive labels or 1
                row_pooled = torch.mean(row, dim=1).detach().cpu().numpy()
                
                id_neg = tree_fns.query(row_pooled, k=[2])
                
                if id_neg[0][
                    0] == 0:  # this means we are getting fps vector TODO: remove fns from the test set before running the algorithm
                    continue
                else:
                    vector_id = id_neg[1][0]
                    # print(id_neg)

                    if vector_id < len(fns_data_pooled):
                        a1 = fns_data_pooled[vector_id]
                    else:
                        continue  # Skip if vector_id is out of bounds

                    a0 = row_pooled
                    a1 = a1
                    sample_space = np.linspace(a0, a1, precision)
                    
                    # print(f"Feature: {feature} | Original vector value: {a0_val} | Closest FN vector value {a1_val} | Sample space: {sample_space[-2]}")
                    
                    x_copy[index] =  torch.from_numpy(sample_space[-1]).to(x_copy.device)
    

    dataset = TensorDataset(x_copy, ytest)

    new_tst_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    preds, metrics = df_dl_predict2(model, new_tst_loader,  device='cuda') #x_copy
    
    

    print(metrics)
    return metrics1, metrics, x_copy
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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
        required=True,
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
        "--output_dir",
        default="",
        type=str,
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

    parser.add_argument('--shap_file', type=str)

    parser.add_argument('--topf', type=int, default=10)

    parser.add_argument('--increase_fn', action='store_true')

    parser.add_argument('--model_type', type=str, default='bert')

    args = parser.parse_args()






    metrics1, metrics, x_copy = shap_attack_mp(args)

    output_dir = os.path.join(args.output_dir, args.task_name)
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = {
        "metrics1": metrics1,
        "metrics": metrics
    }

    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)
