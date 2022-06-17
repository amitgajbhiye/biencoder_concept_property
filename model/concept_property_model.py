import logging

from torch import logit, nn
from torch.nn.functional import normalize
from transformers import (
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    DebertaForSequenceClassification,
)

log = logging.getLogger(__name__)

MODEL_CLASS = {
    "bert-base-uncased": (BertForSequenceClassification, 103),
    "bert-large-uncased": (BertForSequenceClassification, 103),
    "roberta-base": (RobertaForSequenceClassification, 50264),
    "roberta-large": (RobertaForSequenceClassification, 50264),
    "deberta-base": (DebertaForSequenceClassification, 50264),
    "deberta-large": (DebertaForSequenceClassification, 50264),
}


class ConceptPropertyModel(nn.Module):
    def __init__(self, model_params):
        super(ConceptPropertyModel, self).__init__()

        self.hf_checkpoint_name = model_params.get("hf_checkpoint_name")

        self.model_class, self.mask_token_id = MODEL_CLASS.get(self.hf_checkpoint_name)

        self._concept_property_encoder = self.model_class.from_pretrained(
            model_params.get("hf_model_path"), num_labels=2
        )

    def forward(self, input_id, attention_mask, token_type_id, label=None):

        model_output = self._concept_property_encoder(
            input_ids=input_id,
            attention_mask=attention_mask,
            token_type_ids=token_type_id,
            labels=label,
        )

        loss, logit = model_output[:2]

        return loss, logit
