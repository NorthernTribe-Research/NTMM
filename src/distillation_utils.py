from __future__ import annotations

import inspect
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as f
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer


def compute_classification_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model, temperature: float, alpha: float, 
                 use_cosine_loss: bool = True, use_mse_loss: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.use_cosine_loss = use_cosine_loss
        self.use_mse_loss = use_mse_loss

        self.teacher_model.to(self.args.device)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False

        self.student_input_keys = set(inspect.signature(self.model.forward).parameters.keys())
        self.teacher_input_keys = set(inspect.signature(self.teacher_model.forward).parameters.keys())

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]

        student_inputs = {k: v for k, v in inputs.items() if k in self.student_input_keys}
        student_inputs["labels"] = labels
        student_outputs = model(**student_inputs)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_inputs = {k: v for k, v in inputs.items() if k in self.teacher_input_keys}
            teacher_inputs.pop("labels", None)
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits

        # Hard label loss (cross-entropy with ground truth)
        hard_loss = f.cross_entropy(student_logits, labels)

        # KL divergence distillation loss (temperature-scaled)
        softened_teacher = f.softmax(teacher_logits / self.temperature, dim=-1)
        softened_student = f.log_softmax(student_logits / self.temperature, dim=-1)
        kl_loss = (
            f.kl_div(softened_student, softened_teacher, reduction="batchmean")
            * (self.temperature ** 2)
        )

        # Optional: Cosine similarity loss on hidden states
        cosine_loss = torch.tensor(0.0, device=student_logits.device)
        if self.use_cosine_loss and hasattr(student_outputs, 'hidden_states') and hasattr(teacher_outputs, 'hidden_states'):
            # Use last hidden state for cosine similarity
            student_hidden = student_outputs.hidden_states[-1].mean(dim=1)  # [batch, hidden]
            teacher_hidden = teacher_outputs.hidden_states[-1].mean(dim=1)  # [batch, hidden]
            cosine_loss = 1.0 - f.cosine_similarity(student_hidden, teacher_hidden, dim=-1).mean()

        # Optional: MSE loss on logits
        mse_loss = torch.tensor(0.0, device=student_logits.device)
        if self.use_mse_loss:
            mse_loss = f.mse_loss(student_logits, teacher_logits)

        # Combined loss with adaptive weighting
        # Primary: KL divergence + hard labels
        # Secondary: cosine similarity + MSE (if enabled)
        distill_loss = kl_loss
        if self.use_cosine_loss:
            distill_loss = distill_loss + 0.1 * cosine_loss
        if self.use_mse_loss:
            distill_loss = distill_loss + 0.05 * mse_loss

        loss = (self.alpha * distill_loss) + ((1.0 - self.alpha) * hard_loss)
        
        if return_outputs:
            return loss, student_outputs
        return loss
