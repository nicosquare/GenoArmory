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
from transformers.models.mistral.modeling_mistral import (
    MistralForSequenceClassification,
    MistralPreTrainedModel,
    MistralModel,
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


class BertForSequenceClassificationAdvV2_mnli(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 3)
        self.classifier2 = nn.Linear(config.hidden_size, 1)

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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

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


class EsmForSequenceClassificationAdvV2_mnli(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.esm = EsmModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 3)
        self.classifier2 = nn.Linear(config.hidden_size, 1)

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
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits1 = self.classifier1(sequence_output)
        logits2 = self.classifier2(sequence_output)
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


class HyenaDNAForSequenceClassificationAdvV2_mnli(HyenaDNAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.hyena = HyenaDNAModel(config)
        self.dropout = nn.Dropout(config.embed_dropout)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False)
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

        logits = self.score(sequence_output)
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
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_output = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
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


class BertForSequenceClassificationAdvV2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 1)

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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

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


class EsmForSequenceClassificationAdvV2(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.esm = EsmModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 1)

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

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits1 = self.classifier1(sequence_output)
        logits2 = self.classifier2(sequence_output)
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


class HyenaDNAForSequenceClassificationAdvV2(HyenaDNAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.hyena = HyenaDNAModel(config)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False)
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

        logits = self.score(sequence_output)
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
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_output = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
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


class MistralForSequenceClassificationAdvV2(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.mistral = MistralModel(config)
        self.dropout = nn.Dropout(getattr(config, "classifier_dropout", 0.1))
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 1)
        self.is_causal = config.is_causal

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids=None,
        head_mask=None,
        use_cache: Optional[bool] = None,
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

        outputs = self.mistral(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.is_causal:
            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError(
                    "Cannot handle batch sizes > 1 if no padding token is defined."
                )
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (
                        torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1)
                        - 1
                    )
                    sequence_lengths = sequence_lengths % input_ids.shape[-1]
                    sequence_lengths = sequence_lengths.to(sequence_output.device)
                else:
                    sequence_lengths = -1

            # Get the hidden states for the last token in each sequence
            pooled_output = sequence_output[
                torch.arange(batch_size, device=sequence_output.device),
                sequence_lengths,
            ]
        else:
            # Use the first token's hidden state if not causal
            pooled_output = sequence_output[:, 0]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)  # [batch_size, 2]
        logits2 = self.classifier2(pooled_output)  # [batch_size, 1]
        prob = torch.sigmoid(logits2)

        loss = None
        if labels is not None:
            attack_labels, orig_labels, simplify_labels, isMR_labels = real_labels(
                labels
            )
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(logits1.view(-1, 2), orig_labels.view(-1))
            loss_fct2 = nn.BCEWithLogitsLoss()
            loss2 = loss_fct2(logits2.view(-1), attack_labels.float().view(-1))
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


class MistralForSequenceClassificationAdvV2_mnli(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.mistral = MistralModel(config)
        self.dropout = nn.Dropout(getattr(config, "classifier_dropout", 0.1))
        self.classifier1 = nn.Linear(config.hidden_size, 3)  # 3 classes for MNLI
        self.classifier2 = nn.Linear(config.hidden_size, 1)
        self.is_causal = config.is_causal

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids=None,
        head_mask=None,
        use_cache: Optional[bool] = None,
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

        outputs = self.mistral(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.is_causal:
            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError(
                    "Cannot handle batch sizes > 1 if no padding token is defined."
                )
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (
                        torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1)
                        - 1
                    )
                    sequence_lengths = sequence_lengths % input_ids.shape[-1]
                    sequence_lengths = sequence_lengths.to(sequence_output.device)
                else:
                    sequence_lengths = -1

            # Get the hidden states for the last token in each sequence
            pooled_output = sequence_output[
                torch.arange(batch_size, device=sequence_output.device),
                sequence_lengths,
            ]
        else:
            # Use the first token's hidden state if not causal
            pooled_output = sequence_output[:, 0]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)  # [batch_size, 3]
        logits2 = self.classifier2(pooled_output)  # [batch_size, 1]
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


class BertForSequenceClassificationAdvV3(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.pooler1 = Pooler(config)
        self.pooler2 = Pooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.classifier3 = nn.Linear(config.hidden_size, 1)

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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

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


class MistralForSequenceClassificationAdvV3(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.mistral = MistralModel(config)
        self.dropout = nn.Dropout(getattr(config, "classifier_dropout", 0.1))
        self.pooler1 = Pooler(config)
        self.pooler2 = Pooler(config)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.classifier3 = nn.Linear(config.hidden_size, 1)
        self.is_causal = config.is_causal

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids=None,
        head_mask=None,
        use_cache: Optional[bool] = None,
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

        outputs = self.mistral(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.is_causal:
            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError(
                    "Cannot handle batch sizes > 1 if no padding token is defined."
                )
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (
                        torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1)
                        - 1
                    )
                    sequence_lengths = sequence_lengths % input_ids.shape[-1]
                    sequence_lengths = sequence_lengths.to(sequence_output.device)
                else:
                    sequence_lengths = -1

            # Get the hidden states for the last token in each sequence
            pooled_output = sequence_output[
                torch.arange(batch_size, device=sequence_output.device),
                sequence_lengths,
            ]
        else:
            # Use the first token's hidden state if not causal
            pooled_output = sequence_output[:, 0]

        pooled_output = self.dropout(pooled_output)
        output2 = self.pooler1(pooled_output)
        output3 = self.pooler2(pooled_output)
        logits1 = self.classifier1(pooled_output)  # [batch_size, 2]
        logits2 = self.classifier2(output2)  # [batch_size, 2]
        logits3 = self.classifier3(output3)  # [batch_size, 1]
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


class HyenaDNAForSequenceClassificationAdvV3(HyenaDNAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.hyena = HyenaDNAModel(config)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False)
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

        logits = self.score(sequence_output)
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
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_output = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
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


class EsmForSequenceClassificationAdvV3(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.esm = EsmModel(config)
        self.pooler1 = Pooler(config)
        self.pooler2 = Pooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.classifier3 = nn.Linear(config.hidden_size, 1)

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

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        output2 = self.pooler1(sequence_output)
        output3 = self.pooler2(sequence_output)
        logits1 = self.classifier1(sequence_output)
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


class BertForSequenceClassificationAdvBase(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 1)

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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
      
        loss = None
        if labels is not None:
            attack_labels, orig_labels = real_labels(labels)
            loss_fct1 = CrossEntropyLoss()
            active_logits = logits1.view(-1, 2)
            active_labels = orig_labels.view(-1)
            loss1 = loss_fct1(active_logits, active_labels)
            loss = loss1

        if not return_dict:
            output = (logits1,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits1,
            hidden_states=outputs[0],
            attentions=outputs.attentions,
        )


class BertForSequenceClassificationAdv(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.classifier3 = nn.Linear(config.hidden_size, 1)

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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        logits3 = self.classifier3(pooled_output)
        prob = torch.sigmoid(logits3)
        logits = logits1.mul(prob) + logits2.mul(1 - prob)
      
        loss = None
        if labels is not None:
            attack_labels, orig_labels = real_labels(labels)
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(logits.view(-1, self.num_labels), orig_labels.view(-1))
            loss_fct2 = nn.BCEWithLogitsLoss()
            loss2 = loss_fct2(logits3.view(-1), attack_labels.float().view(-1))
            loss3 = 1 / torch.norm(logits1 - logits2)
            loss = loss1 + loss2 + loss3

        if not return_dict:
            output = (logits, prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs[0],
            attentions=outputs.attentions,
        )


class BertForSequenceClassificationAdvNew(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.classifier3 = nn.Linear(config.hidden_size, 1)

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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        logits3 = self.classifier3(pooled_output)
        prob = torch.sigmoid(logits3)
        logits = logits1.mul(prob) + logits2.mul(1 - prob)
      
        loss = None
        if labels is not None:
            attack_labels, orig_labels = real_labels(labels)
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(logits.view(-1, self.num_labels), orig_labels.view(-1))
            loss_fct2 = nn.BCEWithLogitsLoss()
            loss2 = loss_fct2(logits3.view(-1), attack_labels.float().view(-1))
            loss3 = 1 / torch.norm(logits1 - logits2)
            loss = loss1 + loss2 + loss3

        if not return_dict:
            output = (logits, prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs[0],
            attentions=outputs.attentions,
        )


class BertForSequenceClassificationRecover(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.classifier3 = nn.Linear(config.hidden_size, 1)

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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        logits3 = self.classifier3(pooled_output)
        prob = torch.sigmoid(logits3)
        logits = logits1.mul(prob) + logits2.mul(1 - prob)
      
        loss = None
        if labels is not None:
            attack_labels, orig_labels = real_labels(labels)
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(logits.view(-1, self.num_labels), orig_labels.view(-1))
            loss_fct2 = nn.BCEWithLogitsLoss()
            loss2 = loss_fct2(logits3.view(-1), attack_labels.float().view(-1))
            loss3 = 1 / torch.norm(logits1 - logits2)
            loss = loss1 + loss2 + loss3

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PrLMForClassificationSvd(AutoModelForSequenceClassification):
    def __init__(self):
        super(PrLMForClassificationSvd, self).__init__()

    def from_pretrained_svd(pretrained_model_name_or_path, from_tf, config, cache_dir):
        model = AutoModelForSequenceClassification.from_pretrained(
            # pretrained_model_name_or_path='bert-base-uncased',
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            from_tf=from_tf,
            config=config,
            cache_dir=cache_dir,
        )
        # model.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        # print(config.num_labels)
        if config.svd_reserve_size != 0:
            u, s, v = torch.svd(model.bert.embeddings.word_embeddings.weight.data)
            s_new = torch.zeros([len(s)])
            for i in range(config.svd_reserve_size):
                s_new[i] = s[i]
            weight_new = torch.matmul(
                torch.matmul(u, torch.diag_embed(s_new)), v.transpose(-2, -1)
            )
            model.bert.embeddings.word_embeddings.weight.data.copy_(weight_new)
            # model.bert.embeddings.word_embeddings.requires_grad_(False)
        return model


class PrLMForClassificationSvdElectra(AutoModelForSequenceClassification):
    def __init__(self):
        super(PrLMForClassificationSvdElectra, self).__init__()

    def from_pretrained_svd(pretrained_model_name_or_path, from_tf, config, cache_dir):
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            from_tf=from_tf,
            config=config,
            cache_dir=cache_dir,
        )
        if config.svd_reserve_size != 0:
            u, s, v = torch.svd(model.electra.embeddings.word_embeddings.weight.data)
            s_new = torch.zeros([len(s)])
            for i in range(config.svd_reserve_size):
                s_new[i] = s[i]
            weight_new = torch.matmul(
                torch.matmul(u, torch.diag_embed(s_new)), v.transpose(-2, -1)
            )
            print(model.electra.embeddings.word_embeddings.weight.data - weight_new)
            model.electra.embeddings.word_embeddings.weight.data.copy_(weight_new)
        return model


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output
