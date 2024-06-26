import torch
from fancy_einsum import einsum


def calculate_load_balancing_loss(
    alpha: float,
    softmax_per_token: torch.Tensor,
    n_tokens_in_each_expert: torch.Tensor,
    use_einsum: bool = False,
):
    """
    Calculates the load balancing loss for the token choice layer.

    :param str alpha: aux loss weigth parameter
    :param torch.Tensor softmax_per_token: tensor of shape (tokens, n_experts)
    :param torch.Tensor tokens_in_each_expert: tensor of shape (n_experts)
    """
    n_tokens, n_experts = softmax_per_token.shape
    assert n_experts == n_tokens_in_each_expert.shape[0]

    per_expert_softmax_sum = torch.mean(softmax_per_token, dim=0)

    if use_einsum:
        dot_product = einsum("i, i ->", per_expert_softmax_sum, n_tokens_in_each_expert)
    else:
        dot_product = torch.dot(per_expert_softmax_sum, n_tokens_in_each_expert)
    load_balancing_loss = alpha * n_experts * dot_product / n_tokens
    return load_balancing_loss


def calculate_z_loss(zloss_weight: float = 0, gate_logits: torch.Tensor = None):
    zloss = torch.logsumexp(gate_logits, dim=0)
    zloss = torch.square(zloss)
    zloss = zloss.mean()
    zloss = zloss_weight * zloss

    return zloss
