import torch

from lizrd.core.misc import LoggingLayer


class GradLoggingLayer(LoggingLayer):
    def log_heavy(self):
        log_dict = super().log_heavy()

        raw_grad_norms = torch.norm(self.logging_cache["raw_grad"], dim=-1)  # (batch_size, max_length)
        norm_grad_norms = torch.norm(self.logging_cache["norm_grad"], dim=-1)  # (batch_size, max_length)
        activation_norms = torch.norm(self.logging_cache["activations"], dim=-1)

        raw_grad_norms_mean = torch.mean(raw_grad_norms)  # avegare over all tokens and batches
        raw_grad_norms_std = torch.std(raw_grad_norms)
        norm_grad_norms_mean = torch.mean(norm_grad_norms)
        norm_grad_norms_std = torch.std(norm_grad_norms)
        activation_norms_mean = torch.mean(activation_norms)
        activation_norms_std = torch.std(activation_norms)

        log_dict["raw_grad_norms/mean"] = raw_grad_norms_mean
        log_dict["raw_grad_norms/std"] = raw_grad_norms_std
        log_dict["norm_grad_norms/mean"] = norm_grad_norms_mean
        log_dict["norm_grad_norms/std"] = norm_grad_norms_std
        log_dict["activation_norms/mean"] = activation_norms_mean
        log_dict["activation_norms/std"] = activation_norms_std

        return log_dict
