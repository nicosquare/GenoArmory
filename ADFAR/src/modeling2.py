from transformers import AutoModel
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModel
from bert_layers import BertForSequenceClassification, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PretrainedConfig, AutoConfig
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
from modeling_hyena import (
    HyenaDNAForSequenceClassification,
    HyenaDNAModel,
    HyenaDNAPreTrainedModel,
)
from transformers.models.esm.modeling_esm import (
    EsmForSequenceClassification,
    EsmModel,
    EsmPreTrainedModel,
)

from typing import List, Optional, Tuple, Union


def real_labels(labels):
    attack_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    orig_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    simplify_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    isMR_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    for i in range(len(labels)):
        if labels[i] > 9:
            attack_labels[i] = 1
            orig_labels[i] = labels[i] - 10
        elif labels[i] > 7:
            orig_labels[i] = labels[i] - 8
        elif labels[i] > 5:
            simplify_labels[i] = 1
            attack_labels[i] = 1
            isMR_labels[i] = 1
            orig_labels[i] = labels[i] - 6
        elif labels[i] > 3:
            simplify_labels[i] = 1
            isMR_labels[i] = 1
            orig_labels[i] = labels[i] - 4
        elif labels[i] > 1:
            attack_labels[i] = 1
            isMR_labels[i] = 1
            orig_labels[i] = labels[i] - 2
        elif labels[i] > -1:
            isMR_labels[i] = 1
            orig_labels[i] = labels[i]
    return attack_labels, orig_labels, simplify_labels, isMR_labels


def real_labels_mnli(labels):
    attack_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    orig_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    simplify_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    for i in range(len(labels)):
        if labels[i] < 3:
            orig_labels[i] = labels[i]
        elif labels[i] < 6:
            orig_labels[i] = labels[i] - 3
            simplify_labels[i] = 1
        elif labels[i] < 9:
            attack_labels[i] = 1
            orig_labels[i] = labels[i] - 6
        elif labels[i] < 12:
            simplify_labels[i] = 1
            attack_labels[i] = 1
            orig_labels[i] = labels[i] - 9
    return attack_labels, orig_labels, simplify_labels


from transformers.activations import get_activation





class HyenaDNAForSequenceClassificationAdvV3(HyenaDNAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.hyena = HyenaDNAModel(config)
        
        self.pooler1 = Pooler(config)
        self.pooler2 = Pooler(config)
        self.dropout = nn.Dropout(config.embed_dropout)
        self.classifier1 = nn.Linear(config.d_model, 2)
        self.classifier2 = nn.Linear(config.d_model, 2)
        self.classifier3 = nn.Linear(config.d_model, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        inference=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.hyena(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1
                ).to(sequence_output.device)
            else:
                sequence_lengths = -1

        pooled_output = sequence_output[
            torch.arange(batch_size, device=sequence_output.device), sequence_lengths
        ]

        pooled_output = self.dropout(pooled_output)
        output2 = self.pooler1(pooled_output)
        output3 = self.pooler2(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(output2)
        logits3 = self.classifier3(output3)
        prob = torch.sigmoid(logits3)
      
        loss = None
        if labels is not None:
            attack_labels, orig_labels, simplify_labels, isMR_labels = real_labels(
                labels
            )
            loss_fct1 = CrossEntropyLoss()
            active_loss1_notattack = attack_labels.view(-1) == 0
            active_loss1_isMR = isMR_labels.view(-1) == 1
            active_loss1 = active_loss1_notattack & active_loss1_isMR
            active_logits1 = logits1.view(-1, 2)[active_loss1]
            active_labels1 = orig_labels.view(-1)[active_loss1]
            loss1 = loss_fct1(active_logits1, active_labels1)
            # active_loss2 = attack_labels.view(-1) == 1
            active_loss2_isattack = attack_labels.view(-1) == 1
            active_loss2_isMR = isMR_labels.view(-1) == 1
            active_loss2 = active_loss2_isattack & active_loss2_isMR
            active_logits2 = logits2.view(-1, 2)[active_loss2]
            active_labels2 = orig_labels.view(-1)[active_loss2]
            loss2 = loss_fct1(active_logits2, active_labels2)
            loss_fct2 = nn.BCEWithLogitsLoss()

            active_loss3 = simplify_labels.view(-1) == 0
            active_logits3 = logits3.view(-1)[active_loss3]
            active_labels3 = attack_labels.float().view(-1)[active_loss3]
            loss3 = loss_fct2(active_logits3, active_labels3)

            # loss3 = loss_fct2(logits3.view(-1), attack_labels.float().view(-1))
            loss = loss1 + loss2 + loss3

        if inference:
            output = (logits1, logits2, prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:
            logits = []
            for i, prob_sentence in enumerate(prob):
                if prob_sentence[0] <= 0.5:
                    logits.append(logits1[i].tolist())
                else:
                    logits.append(logits2[i].tolist())
            logits = torch.Tensor(logits).cuda()

            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits1,
            hidden_states=outputs[0],
            # attentions=outputs.attentions,
        )


class HyenaDNAForSequenceClassificationAdvV2(HyenaDNAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.hyena = HyenaDNAModel(config)
        self.dropout = nn.Dropout(config.embed_dropout)
        self.classifier1 = nn.Linear(config.d_model, 2)
        self.classifier2 = nn.Linear(config.d_model, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        inference=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.hyena(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1
                ).to(sequence_output.device)
            else:
                sequence_lengths = -1

        pooled_output = sequence_output[
            torch.arange(batch_size, device=sequence_output.device), sequence_lengths
        ]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        prob = torch.sigmoid(logits2)
        loss = None
        if labels is not None:
            attack_labels, orig_labels, simplify_labels, isMR_labels = real_labels(
                labels
            )
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(logits1.view(-1, 2), orig_labels.view(-1))
            loss_fct2 = nn.BCEWithLogitsLoss()
            active_loss2 = simplify_labels.view(-1) == 0
            active_logits2 = logits2.view(-1)[active_loss2]
            active_labels2 = attack_labels.float().view(-1)[active_loss2]
            loss2 = loss_fct2(active_logits2, active_labels2)
            loss = loss1 + loss2

        if inference:
            output = (logits1, prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:
            output = (logits1,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits1,
            hidden_states=outputs[0],
            # attentions=outputs.attentions,
        )


class HyenaDNAForSequenceClassificationAdvV2_mnli(HyenaDNAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.hyena = HyenaDNAModel(config)
        self.dropout = nn.Dropout(config.embed_dropout)
        self.classifier1 = nn.Linear(config.d_model, 3)
        self.classifier2 = nn.Linear(config.d_model, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        inference=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.hyena(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1
                ).to(sequence_output.device)
            else:
                sequence_lengths = -1

        pooled_output = sequence_output[
            torch.arange(batch_size, device=sequence_output.device), sequence_lengths
        ]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        prob = torch.sigmoid(logits2)
        loss = None
        if labels is not None:
            attack_labels, orig_labels, simplify_labels = real_labels_mnli(labels)
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(logits1.view(-1, 3), orig_labels.view(-1))
            loss_fct2 = nn.BCEWithLogitsLoss()
            active_loss2 = simplify_labels.view(-1) == 0
            active_logits2 = logits2.view(-1)[active_loss2]
            active_labels2 = attack_labels.float().view(-1)[active_loss2]
            loss2 = loss_fct2(active_logits2, active_labels2)
            loss = loss1 + loss2

        if inference:
            output = (logits1, prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:
            output = (logits1,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits1,
            hidden_states=outputs[0],
            # attentions=outputs.attentions,
        )

