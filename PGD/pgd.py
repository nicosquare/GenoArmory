# %%
import os
import torch as ch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from robustness.bert_layers import BertForSequenceClassification
import sklearn
import numpy as np
import transformers
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
import argparse
import torch
import random

from genoarmory.PGD.robustness.datasets import DNA
from genoarmory.PGD.robustness.model_utils import make_and_restore_model

ATTACK_EPS = 0.5
ATTACK_STEPSIZE = 0.1
ATTACK_STEPS = 10


kwargs = {
    'constraint':'2', # use L2-PGD
    'eps': ATTACK_EPS, # L2 radius around original image
    'step_size': ATTACK_STEPSIZE,
    'iterations': ATTACK_STEPS,
    'do_tqdm': True,
}



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

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)







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
    "--model_type",
    default='',
    type=str,
)

parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="Path to save attack results",
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
    "--worker", default=16, type=int, help="Number of Worker",
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

parser.add_argument('--n_gpu', type=int, default=1)

parser.add_argument('--num_label', type=int, default=2)

args = parser.parse_args()

set_seed(args)

config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_label,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
        trust_remote_code=True,
    )
if args.model_type == 'nt':
    config.token_dropout = False

tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None,
    trust_remote_code=True,
)
if args.model_type == 'bert':
    model = BertForSequenceClassification.from_pretrained(
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
ds = DNA(args)



# %%

model = make_and_restore_model(args=args, arch=model, dataset=ds)
#model.eval()

test_loader = ds.make_loaders(args, tokenizer, workers=args.worker, batch_size=args.batch_size)


# Initialize variables for accuracy calculation
total_correct = 0
total_samples = 0
all_metrics = []

for batch in test_loader:
    # Prepare input and labels
    if args.model_type == 'hyena':
        im = {"input_ids": batch[0], "labels": batch[-1]}
    else:
        im = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[-1]}
    label = batch[-1]

    # Generate adversarial examples
    _, im_adv = model(im, label, make_adv=True, **kwargs)

    # Get predictions
    pred, _ = model(im_adv)
    label_pred = torch.argmax(pred.logits, dim=1).to(label.device)

    # Ensure labels are on the same device as predictions
    logits = pred.logits.to(label.device)

    # Compute accuracy
    total_correct += (label_pred == label).sum().item()
    total_samples += label.size(0)

    # Compute additional metrics
    metrics = compute_metrics((logits.detach().numpy(), label.detach().numpy()))
    all_metrics.append(metrics)
    print(f"Batch Metrics: {metrics}")

# Final accuracy computation
accuracy = total_correct / total_samples

# Compute average metrics
avg_metrics = {key: sum(d[key] for d in all_metrics) / len(all_metrics) for key in all_metrics[0]}
# Format output
output_text = f"Final Accuracy: {accuracy:.4f}\nAverage Metrics: {avg_metrics}"

# Print results
print(output_text)

# Save to file
os.makedirs(args.output_dir, exist_ok=True)
output_file = os.path.join(args.output_dir, "results.txt")
with open(output_file, "w") as f:
    f.write(output_text)

