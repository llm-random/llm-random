import opt_einsum
import torch
import torch.nn

from lizrd.core import nn


def einsum(subscript, *operands, use_opt_einsum=False, **kwargs):
    if use_opt_einsum:
        return opt_einsum.contract(subscript, *operands, **kwargs)
    else:
        return torch.einsum(subscript, *operands, **kwargs)


def check_layer_funs(*layer_funs):
    for layer_fun in layer_funs:
        if isinstance(layer_fun, nn.Module):
            raise TypeError(
                "Expected layer function/lambda, got nn.Module: {}".format(
                    type(layer_fun)
                )
            )


def stop_gradient(x):
    return x.detach()


def get_default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def default(x, d):
    return x if x is not None else d


def print_available_gpus():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print("Found {} GPU(s)".format(count))
        for i in range(count):
            print("GPU {}: {}".format(i, torch.cuda.get_device_name(i)))


def are_state_dicts_the_same(
    model_state_dict_1: dict, model_state_dict_2: dict
) -> bool:
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}")
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for (k_1, v_1), (k_2, v_2) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: {v_1} vs {v_2}")
            return False
    return True


def get_neuron_magnitudes(
    lin1_weight: torch.Tensor, lin2_weight: torch.Tensor
) -> torch.Tensor:
    weights1 = torch.sqrt(einsum("f m -> f", lin1_weight**2))
    weights2 = torch.sqrt(einsum("m f -> f", lin2_weight**2))

    return (weights1 * weights2).flatten()


def get_split_neuron_magnitudes(
    lin1_weight: torch.Tensor, lin2_weight: torch.Tensor
) -> torch.Tensor:
    """
    Returns the magnitude of the matrix formed by the concatenation of the two weight matrices.
    """
    weights1 = torch.sqrt(einsum("f m -> f", lin1_weight**2))
    weights2 = torch.sqrt(einsum("m f -> f", lin2_weight**2))

    # return concatenation
    return torch.cat((weights1**2, weights2**2), dim=0).flatten()


def get_mixed_neuron_magnitudes(
    lin1_weight: torch.Tensor, lin2_weight: torch.Tensor
) -> torch.Tensor:
    """
    Returns magnitudes of "nerons" formed by random combinations of rows/cols
    """
    weights1 = torch.sqrt(einsum("f m -> f", lin1_weight**2))
    weights2 = torch.sqrt(einsum("m f -> f", lin2_weight**2))

    weights1 = weights1.flatten()
    weights2 = weights2.flatten()
    weights1 = weights1.flip(0)
    return weights1 * weights2


def get_dmodel_magnitudes(
    lin1_weight: torch.Tensor, lin2_weight: torch.Tensor
) -> torch.Tensor:
    """
    Aggregate by dmodel instead of dff
    """
    weights1 = torch.sqrt(einsum("f m -> m", lin1_weight**2))
    weights2 = torch.sqrt(einsum("m f -> m", lin2_weight**2))

    return (weights1 * weights2).flatten()


def resolve_activation_name(activation: str) -> torch.nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return torch.nn.GELU()
    elif activation == "silu":
        return torch.nn.SiLU()
    elif activation == "softmax":
        return torch.nn.Softmax()
    else:
        raise ValueError(f"Unrecognized activation: {activation}")


def propagate_forward_pass_cache(module: torch.nn.Module, forward_pass_cache=None):
    """
    This function propagates the cache from the module to all its children.
    """
    if forward_pass_cache is None:
        forward_pass_cache = dict()
    module.forward_pass_cache = forward_pass_cache
    for child in module.children():
        propagate_forward_pass_cache(child, forward_pass_cache)
