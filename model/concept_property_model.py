import logging
import torch
from torch import nn
from torch.nn import Dropout
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

        self._classifier = nn.Sequential(
            nn.Dropout(self.dropout_prob),
            nn.Linear(
                4 * self._concept_encoder.config.hidden_size, param.get("ff1_out_dim")
            ),
            nn.ReLU(),
            nn.Linear(
                param.get("ff1_out_dim"), self._concept_encoder.config.num_labels
            ),
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

        # TODO AVG and MAX and MIn Over seq_len dimension and concatenate to feed to classifier

        concept_last_hidden_states, concept_cls = (
            concept_output.get("last_hidden_state"),
            concept_output.get("pooler_output"),
        )

        property_last_hidden_states, property_cls = (
            property_output.get("last_hidden_state"),
            property_output.get("pooler_output"),
        )

        # v_sum = torch.add(concept_cls, property_cls)

        concept_cls = concept_cls * concept_attention_mask.unsqueeze(1).transpose(2, 1)
        property_cls = property_cls * property_attention_mask.unsqueeze(1).transpose(
            2, 1
        )

        v_sub = torch.sub(concept_cls, property_cls)
        v_hadamard = torch.mul(concept_cls, property_cls)

        v = torch.cat((concept_cls, property_cls, v_sub, v_hadamard), dim=-1)

        logits = self._classifier(v)

        probabilities = nn.functional.softmax(logits, dim=-1)

        preds = torch.argmax(probabilities, dim=1)

        log.info(
            f"Dimensions v: {v.shape}, logits: {logits.shape}, probabilities: {probabilities.shape}, preds: {preds.shape}"
        )

        return logits, probabilities, preds

