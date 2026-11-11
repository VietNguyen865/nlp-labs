"""Section 13: task-specific classification head over a pretrained encoder."""

import torch
from torch import nn
from transformers import AutoModel


class PretrainedClassificationHead(nn.Module):
    """Encode a sequence, take its [CLS] representation, and predict 2 classes.

    This module implements only the model/head component. Dataset handling,
    training, experiments, metrics, and visualizations belong in the notebooks.
    """

    def __init__(self, model_name: str = "prajjwal1/bert-tiny", num_labels: int = 2,
                 dropout: float = 0.1, encoder: nn.Module | None = None,
                 local_files_only: bool = False):
        super().__init__()
        self.encoder = (encoder if encoder is not None else
                        AutoModel.from_pretrained(model_name, local_files_only=local_files_only))
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        encoder_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        encoded = self.encoder(**encoder_inputs)
        cls_representation = encoded.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_representation))
        return logits

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        """Convert raw class logits to class probabilities for inference."""
        return torch.softmax(logits, dim=-1)
