import logging

import torch
from torch import nn
from torch.nn.functional import normalize
from transformers import BertModel

log = logging.getLogger(__name__)


class ConceptPropertyModel(nn.Module):
    def __init__(self, param):
        super(ConceptPropertyModel, self).__init__()

        # self._concept_encoder = BertModel.from_pretrained("bert-base-uncased")
        # self._property_encoder = BertModel.from_pretrained("bert-base-uncased")

        self._concept_encoder = BertModel.from_pretrained(
            param.get("hf_checkpoint_name")
        )
        self._property_encoder = BertModel.from_pretrained(
            param.get("hf_checkpoint_name")
        )

        self.dropout_prob = param.get("dropout_prob")
        self.strategy = param.get("vector_strategy")

        self.multiplier = {
            "max_avg_pool": 4,
            "cls_sub_mul": 4,
            "mean_only": 2,
            "cls_add_sub_abs": 5,
            "dot_product": 1,
        }

        self.inp_dim = (
            self.multiplier.get(self.strategy)
            * self._concept_encoder.config.hidden_size
        )

        self.out_dim = (
            self.multiplier.get(self.strategy)
            * self._concept_encoder.config.hidden_size
        ) // 2

        self._classifier = nn.Sequential(
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.inp_dim, self.out_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.out_dim, 1),
        )

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

        if self.strategy == "max_avg_pool":

            v_concept_avg = torch.sum(
                concept_last_hidden_states
                * concept_attention_mask.unsqueeze(1).transpose(2, 1),
                dim=1,
            ) / torch.sum(concept_attention_mask, dim=1, keepdim=True)

            v_property_avg = torch.sum(
                property_last_hidden_states
                * property_attention_mask.unsqueeze(1).transpose(2, 1),
                dim=1,
            ) / torch.sum(property_attention_mask, dim=1, keepdim=True)

            v_concept_max, _ = self.replace_masked(
                concept_last_hidden_states, concept_attention_mask, -1e7
            ).max(dim=1)
            v_property_max, _ = self.replace_masked(
                property_last_hidden_states, property_attention_mask, -1e7
            ).max(dim=1)

            v = torch.cat(
                [v_concept_avg, v_concept_max, v_property_avg, v_property_max], dim=1
            )

            logits = self._classifier(v)

            return logits

        elif self.strategy == "mean_only":
            v_concept_avg = torch.sum(
                concept_last_hidden_states
                * concept_attention_mask.unsqueeze(1).transpose(2, 1),
                dim=1,
            ) / torch.sum(concept_attention_mask, dim=1, keepdim=True)

            v_concept_avg = normalize(v_concept_avg, p=2, dim=1)
            log.info(f"Shape of normalised v_concept_avg :{v_concept_avg.shape}")

            v_property_avg = torch.sum(
                property_last_hidden_states
                * property_attention_mask.unsqueeze(1).transpose(2, 1),
                dim=1,
            ) / torch.sum(property_attention_mask, dim=1, keepdim=True)

            v = (
                (v_concept_avg * v_property_avg)
                .sum(-1)
                .reshape(v_concept_avg.shape[0], 1)
            )

            return v

        elif self.strategy == "cls_sub_mul":

            v_sub = torch.sub(concept_cls, property_cls)
            v_hadamard = torch.mul(concept_cls, property_cls)

            v = torch.cat((concept_cls, property_cls, v_sub, v_hadamard), dim=1)

            logits = self._classifier(v)

            return logits

        elif self.strategy == "cls_add_sub_abs":

            v_add = torch.add((concept_cls, property_cls))
            v_sub = torch.sub(concept_cls, property_cls)
            v_abs = torch.abs(v_sub)

            v = torch.cat((concept_cls, property_cls, v_add, v_sub, v_abs), dim=1)

            logits = self._classifier(v)

            return logits

        elif self.strategy == "dot_product":
            # The dot product of concept property cls vector will be a scalar.

            concept_cls = normalize(concept_cls, p=2, dim=1)

            v = (concept_cls * property_cls).sum(-1).reshape(concept_cls.shape[0], 1)

            return v

    def replace_masked(self, tensor, mask, value):
        """
        Replace the all the values of vectors in 'tensor' that are masked in
        'masked' by 'value'.
        Args:
            tensor: The tensor in which the masked vectors must have their values
                replaced.
            mask: A mask indicating the vectors which must have their values
                replaced.
            value: The value to place in the masked vectors of 'tensor'.
        Returns:
            A new tensor of the same size as 'tensor' where the values of the
            vectors masked in 'mask' were replaced by 'value'.
        """
        mask = mask.unsqueeze(1).transpose(2, 1)
        reverse_mask = 1.0 - mask
        values_to_add = value * reverse_mask
        return tensor * mask + values_to_add
