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
