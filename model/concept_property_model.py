import logging

import torch
from torch import nn
from torch.nn.functional import normalize
from transformers import BertModel

log = logging.getLogger(__name__)


class ConceptPropertyModel(nn.Module):
    def __init__(self, model_params):
        super(ConceptPropertyModel, self).__init__()

        # self._concept_encoder = BertModel.from_pretrained("bert-base-uncased")
        # self._property_encoder = BertModel.from_pretrained("bert-base-uncased")

        self._concept_encoder = BertModel.from_pretrained(
            model_params.get("hf_model_path")
        )
        self._property_encoder = BertModel.from_pretrained(
            model_params.get("hf_model_path")
        )

        self.dropout_prob = model_params.get("dropout_prob")
        self.strategy = model_params.get("vector_strategy")

    def forward(
        self,
        concept_input_id,
        concept_attention_mask,
        property_input_id,
        property_attention_mask,
    ):

        concept_output = self._concept_encoder(
            input_ids=concept_input_id, attention_mask=concept_attention_mask
        )

        property_output = self._property_encoder(
            input_ids=property_input_id, attention_mask=property_attention_mask
        )

        concept_last_hidden_states, concept_cls = (
            concept_output.get("last_hidden_state"),
            concept_output.get("pooler_output"),
        )

        property_last_hidden_states, property_cls = (
            property_output.get("last_hidden_state"),
            property_output.get("pooler_output"),
        )

        if self.strategy == "mean":

            # The dot product of the average of the last hidden states of the concept and property hidden states.

            v_concept_avg = torch.sum(
                concept_last_hidden_states
                * concept_attention_mask.unsqueeze(1).transpose(2, 1),
                dim=1,
            ) / torch.sum(concept_attention_mask, dim=1, keepdim=True)

            # Normalising concept vectors
            v_concept_avg = normalize(v_concept_avg, p=2, dim=1)

            v_property_avg = torch.sum(
                property_last_hidden_states
                * property_attention_mask.unsqueeze(1).transpose(2, 1),
                dim=1,
            ) / torch.sum(property_attention_mask, dim=1, keepdim=True)

            logits = (
                (v_concept_avg * v_property_avg)
                .sum(-1)
                .reshape(v_concept_avg.shape[0], 1)
            )

            return v_concept_avg, v_property_avg, logits

        elif self.strategy == "cls":
            # The dot product of concept property cls vectors.

            # Normalising concept vectors
            concept_cls = normalize(concept_cls, p=2, dim=1)

            logits = (
                (concept_cls * property_cls).sum(-1).reshape(concept_cls.shape[0], 1)
            )

            return concept_cls, property_cls, logits
