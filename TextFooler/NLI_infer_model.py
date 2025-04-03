import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, BertConfig, AutoConfig
from NLI_inf_data import NLIDataset_Hyena, NLIDataset_NT, NLIDataset_BERT

class NLI_infer_Hyena(nn.Module):
    def __init__(self, pretrained_dir, nclasses, max_seq_length=128, batch_size=32):
        super(NLI_infer_Hyena, self).__init__()
        self.model_config = AutoConfig.from_pretrained(
            pretrained_dir, num_labels=nclasses, trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_dir, config=self.model_config, trust_remote_code=True
        ).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_Hyena(
            pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size
        )
        self.tokenizer = self.dataset.tokenizer

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        for input_ids,_ in dataloader:
            input_ids = input_ids.cuda()

            with torch.no_grad():
                # Get logits from the model
                logits = self.model(input_ids).logits
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

class NLI_infer_NT(nn.Module):
    def __init__(self, pretrained_dir, nclasses, max_seq_length=128, batch_size=32):
        super(NLI_infer_NT, self).__init__()
        self.model_config = AutoConfig.from_pretrained(
            pretrained_dir, num_labels=nclasses
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_dir, config=self.model_config, trust_remote_code=True
        ).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_NT(
            pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size
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



class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model_config = BertConfig.from_pretrained(pretrained_dir, num_labels=nclasses)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_dir, config=self.model_config, trust_remote_code=True).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)
        self.tokenizer = self.dataset.tokenizer

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

