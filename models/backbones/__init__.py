from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    BertModel,
)

import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


class BertModelWithCustomLossFunction(nn.Module):
    def __init__(self, exp_dict):
        super(BertModelWithCustomLossFunction, self).__init__()
        self.num_labels = exp_dict["dataset"]["num_labels"]
        self.bert = BertModel.from_pretrained(
            exp_dict["model"]["backbone"], num_labels=self.num_labels
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(1024, self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        output = self.dropout(outputs.pooler_output)
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            # you can define any loss function here yourself
            # see https://pytorch.org/docs/stable/nn.html#loss-functions for an overview
            loss_fct = nn.CrossEntropyLoss()
            # next, compute the loss based on logits + ground-truth labels
            loss = loss_fct(logits.view(-1, self.num_labels), labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def get_backbone(exp_dict):
    if exp_dict["exp_type"] == "gpt3mix":
        backbone = BertModelWithCustomLossFunction(exp_dict)
        return backbone

    if exp_dict["model"]["backbone"] in [
        "distilbert-base-uncased",
        "bert-large-uncased",
        "bert-base-uncased",
    ]:
        backbone = AutoModelForSequenceClassification.from_pretrained(
            exp_dict["model"]["backbone"], num_labels=exp_dict["dataset"]["num_labels"]
        )
        return backbone
    raise ValueError(f"backbone: {exp_dict['model']['backbone']} not supported")
