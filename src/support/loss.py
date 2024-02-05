import torch


LossDict = dict[str, torch.Tensor]
RunningLossDict = dict[str, float]
LossWeightDict = dict[str, float]


def update_losses_dict(current_losses: LossDict, new_losses: LossDict) -> LossDict:
    overlapping_keys = current_losses.keys() & new_losses.keys()
    if len(overlapping_keys) > 0:
        raise ValueError(f"Keys overlap: {overlapping_keys}")
    return {**current_losses, **new_losses}
