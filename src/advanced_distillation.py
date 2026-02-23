"""
Advanced Distillation Techniques for NTMM

This module implements cutting-edge distillation methods:
- Embedding distillation
- Attention transfer
- Layer-wise distillation
- Contrastive learning
- Dynamic temperature scaling

Copyright (c) 2026 NorthernTribe Research
"""

from __future__ import annotations

import inspect
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer


class AdvancedDistillationTrainer(Trainer):
    """
    Advanced distillation trainer with multiple knowledge transfer mechanisms.
    
    Features:
    - Embedding distillation: Transfer token embedding knowledge
    - Attention transfer: Distill attention patterns
    - Layer-wise distillation: Multi-layer knowledge transfer
    - Contrastive learning: Better representation learning
    - Dynamic temperature: Adaptive temperature scaling
    """
    
    def __init__(
        self,
        *args,
        teacher_model,
        temperature: float = 3.0,
        alpha: float = 0.5,
        use_embedding_distillation: bool = True,
        use_attention_transfer: bool = True,
        use_layer_distillation: bool = True,
        use_contrastive_loss: bool = False,
        use_dynamic_temperature: bool = False,
        embedding_weight: float = 0.1,
        attention_weight: float = 0.1,
        layer_weight: float = 0.1,
        contrastive_weight: float = 0.05,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Feature flags
        self.use_embedding_distillation = use_embedding_distillation
        self.use_attention_transfer = use_attention_transfer
        self.use_layer_distillation = use_layer_distillation
        self.use_contrastive_loss = use_contrastive_loss
        self.use_dynamic_temperature = use_dynamic_temperature
        
        # Loss weights
        self.embedding_weight = embedding_weight
        self.attention_weight = attention_weight
        self.layer_weight = layer_weight
        self.contrastive_weight = contrastive_weight
        
        # Setup teacher model
        self.teacher_model.to(self.args.device)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False
        
        # Get input keys
        self.student_input_keys = set(inspect.signature(self.model.forward).parameters.keys())
        self.teacher_input_keys = set(inspect.signature(self.teacher_model.forward).parameters.keys())
        
        # Dynamic temperature tracking
        self.step_count = 0
        self.temperature_history = []
    
    def compute_embedding_distillation_loss(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Distill embedding knowledge using MSE loss.
        
        Args:
            student_embeddings: Student token embeddings [batch, seq, hidden]
            teacher_embeddings: Teacher token embeddings [batch, seq, hidden]
            
        Returns:
            Embedding distillation loss
        """
        # Project student embeddings to teacher dimension if needed
        if student_embeddings.shape[-1] != teacher_embeddings.shape[-1]:
            # Use mean pooling for dimension reduction
            student_embeddings = F.adaptive_avg_pool1d(
                student_embeddings.transpose(1, 2),
                teacher_embeddings.shape[-1]
            ).transpose(1, 2)
        
        # MSE loss on embeddings
        return F.mse_loss(student_embeddings, teacher_embeddings)
    
    def compute_attention_transfer_loss(
        self,
        student_attentions: tuple,
        teacher_attentions: tuple
    ) -> torch.Tensor:
        """
        Transfer attention patterns from teacher to student.
        
        Args:
            student_attentions: Tuple of student attention matrices
            teacher_attentions: Tuple of teacher attention matrices
            
        Returns:
            Attention transfer loss
        """
        if not student_attentions or not teacher_attentions:
            return torch.tensor(0.0, device=self.args.device)
        
        total_loss = 0.0
        num_layers = min(len(student_attentions), len(teacher_attentions))
        
        # Match layers (use uniform sampling if different number of layers)
        teacher_indices = np.linspace(0, len(teacher_attentions) - 1, num_layers, dtype=int)
        
        for i, teacher_idx in enumerate(teacher_indices):
            student_attn = student_attentions[i]  # [batch, heads, seq, seq]
            teacher_attn = teacher_attentions[teacher_idx]
            
            # Average over heads
            student_attn_avg = student_attn.mean(dim=1)  # [batch, seq, seq]
            teacher_attn_avg = teacher_attn.mean(dim=1)
            
            # MSE loss on attention patterns
            total_loss += F.mse_loss(student_attn_avg, teacher_attn_avg)
        
        return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)
    
    def compute_layer_distillation_loss(
        self,
        student_hidden_states: tuple,
        teacher_hidden_states: tuple
    ) -> torch.Tensor:
        """
        Layer-wise distillation of hidden representations.
        
        Args:
            student_hidden_states: Tuple of student hidden states
            teacher_hidden_states: Tuple of teacher hidden states
            
        Returns:
            Layer-wise distillation loss
        """
        if not student_hidden_states or not teacher_hidden_states:
            return torch.tensor(0.0, device=self.args.device)
        
        total_loss = 0.0
        num_layers = min(len(student_hidden_states), len(teacher_hidden_states))
        
        # Match layers
        teacher_indices = np.linspace(0, len(teacher_hidden_states) - 1, num_layers, dtype=int)
        
        for i, teacher_idx in enumerate(teacher_indices):
            student_hidden = student_hidden_states[i]  # [batch, seq, hidden]
            teacher_hidden = teacher_hidden_states[teacher_idx]
            
            # Mean pooling over sequence
            student_pooled = student_hidden.mean(dim=1)  # [batch, hidden]
            teacher_pooled = teacher_hidden.mean(dim=1)
            
            # Project if dimensions don't match
            if student_pooled.shape[-1] != teacher_pooled.shape[-1]:
                student_pooled = F.adaptive_avg_pool1d(
                    student_pooled.unsqueeze(1),
                    teacher_pooled.shape[-1]
                ).squeeze(1)
            
            # Cosine similarity loss
            cosine_sim = F.cosine_similarity(student_pooled, teacher_pooled, dim=-1)
            total_loss += (1.0 - cosine_sim).mean()
        
        return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)
    
    def compute_contrastive_loss(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Contrastive learning loss for better representation learning.
        
        Uses SimCLR-style contrastive loss to align student and teacher representations.
        
        Args:
            student_hidden: Student representations [batch, hidden]
            teacher_hidden: Teacher representations [batch, hidden]
            labels: Class labels [batch]
            temperature: Contrastive temperature
            
        Returns:
            Contrastive loss
        """
        batch_size = student_hidden.shape[0]
        
        # Normalize representations
        student_norm = F.normalize(student_hidden, dim=-1)
        teacher_norm = F.normalize(teacher_hidden, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(student_norm, teacher_norm.T) / temperature
        
        # Create positive mask (same class)
        labels_expanded = labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        
        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(0)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Mean over positive pairs
        positive_pairs = positive_mask.sum(dim=1)
        positive_pairs = torch.clamp(positive_pairs, min=1.0)  # Avoid division by zero
        
        loss = -(log_prob * positive_mask).sum(dim=1) / positive_pairs
        return loss.mean()
    
    def compute_dynamic_temperature(self, epoch: int, total_epochs: int) -> float:
        """
        Compute dynamic temperature that decreases over training.
        
        Higher temperature early (more knowledge transfer)
        Lower temperature later (focus on hard labels)
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            
        Returns:
            Dynamic temperature value
        """
        # Cosine annealing from initial temperature to 1.0
        min_temp = 1.0
        max_temp = self.temperature
        progress = epoch / max(total_epochs, 1)
        temp = min_temp + 0.5 * (max_temp - min_temp) * (1 + np.cos(np.pi * progress))
        return temp
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute advanced distillation loss with multiple knowledge transfer mechanisms.
        """
        labels = inputs["labels"]
        
        # Forward pass through student
        student_inputs = {k: v for k, v in inputs.items() if k in self.student_input_keys}
        student_inputs["labels"] = labels
        student_inputs["output_hidden_states"] = True
        student_inputs["output_attentions"] = self.use_attention_transfer
        
        student_outputs = model(**student_inputs)
        student_logits = student_outputs.logits
        
        # Forward pass through teacher
        with torch.no_grad():
            teacher_inputs = {k: v for k, v in inputs.items() if k in self.teacher_input_keys}
            teacher_inputs.pop("labels", None)
            teacher_inputs["output_hidden_states"] = True
            teacher_inputs["output_attentions"] = self.use_attention_transfer
            
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits
        
        # 1. Hard label loss (cross-entropy)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 2. KL divergence distillation loss
        current_temp = self.temperature
        if self.use_dynamic_temperature and hasattr(self.state, 'epoch'):
            current_temp = self.compute_dynamic_temperature(
                int(self.state.epoch),
                int(self.args.num_train_epochs)
            )
            self.temperature_history.append(current_temp)
        
        softened_teacher = F.softmax(teacher_logits / current_temp, dim=-1)
        softened_student = F.log_softmax(student_logits / current_temp, dim=-1)
        kl_loss = (
            F.kl_div(softened_student, softened_teacher, reduction="batchmean")
            * (current_temp ** 2)
        )
        
        # 3. Embedding distillation loss
        embedding_loss = torch.tensor(0.0, device=student_logits.device)
        if self.use_embedding_distillation and hasattr(student_outputs, 'hidden_states'):
            student_embeddings = student_outputs.hidden_states[0]  # First layer (embeddings)
            teacher_embeddings = teacher_outputs.hidden_states[0]
            embedding_loss = self.compute_embedding_distillation_loss(
                student_embeddings, teacher_embeddings
            )
        
        # 4. Attention transfer loss
        attention_loss = torch.tensor(0.0, device=student_logits.device)
        if self.use_attention_transfer and hasattr(student_outputs, 'attentions'):
            attention_loss = self.compute_attention_transfer_loss(
                student_outputs.attentions,
                teacher_outputs.attentions
            )
        
        # 5. Layer-wise distillation loss
        layer_loss = torch.tensor(0.0, device=student_logits.device)
        if self.use_layer_distillation and hasattr(student_outputs, 'hidden_states'):
            layer_loss = self.compute_layer_distillation_loss(
                student_outputs.hidden_states,
                teacher_outputs.hidden_states
            )
        
        # 6. Contrastive loss
        contrastive_loss = torch.tensor(0.0, device=student_logits.device)
        if self.use_contrastive_loss and hasattr(student_outputs, 'hidden_states'):
            student_hidden = student_outputs.hidden_states[-1].mean(dim=1)
            teacher_hidden = teacher_outputs.hidden_states[-1].mean(dim=1)
            contrastive_loss = self.compute_contrastive_loss(
                student_hidden, teacher_hidden, labels
            )
        
        # Combine all losses
        distill_loss = (
            kl_loss +
            self.embedding_weight * embedding_loss +
            self.attention_weight * attention_loss +
            self.layer_weight * layer_loss +
            self.contrastive_weight * contrastive_loss
        )
        
        # Final weighted combination
        loss = (self.alpha * distill_loss) + ((1.0 - self.alpha) * hard_loss)
        
        # Log individual losses (optional)
        if self.step_count % 100 == 0:
            self.log({
                "kl_loss": kl_loss.item(),
                "embedding_loss": embedding_loss.item(),
                "attention_loss": attention_loss.item(),
                "layer_loss": layer_loss.item(),
                "contrastive_loss": contrastive_loss.item(),
                "temperature": current_temp,
            })
        
        self.step_count += 1
        
        if return_outputs:
            return loss, student_outputs
        return loss


def compute_classification_metrics(eval_pred) -> Dict[str, float]:
    """Compute classification metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }
