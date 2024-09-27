from unittest.mock import MagicMock

import pytest

from research.grad_norm.build import get_grad_modif_fn, get_grad_modif_placement
from research.grad_norm.modules import BlockGradModifPlacement
from research.grad_norm.modules.grad_norm import (
    GradientActivationNormLayer,
    GradientScaleNormLayer,
    GradientSTDNormLayerV1,
    GradientSTDNormLayerV2,
    GradientSTDNormLayerV3,
)


def set_attrs_mock(m: MagicMock, d: dict):
    m.__getitem__.side_effect = d.__getitem__
    m.get.side_effect = d.get

    for k, v in d.items():
        if isinstance(v, dict):
            new_mock = MagicMock()
            set_attrs_mock(new_mock, v)
            setattr(m, k, new_mock)
        if isinstance(v, (list, str, type(None))):
            setattr(m, k, v)
            pass
        else:
            raise ValueError("Invalid value type")
    return m


@pytest.fixture
def args_mock(request):
    return set_attrs_mock(MagicMock(), request.param)


@pytest.mark.parametrize(
    "args_mock",
    [
        {
            "grad_modif_placement": ["post_attn", "post_ff", "post_ff_norm"],
        }
    ],
    indirect=True,
)
def test_get_grad_modif_placement(args_mock):
    assert get_grad_modif_placement(args_mock) == BlockGradModifPlacement.from_list(
        ["post_attn", "post_ff", "post_ff_norm"]
    )


@pytest.mark.parametrize(
    ("args_mock", "expected_layer_type", "expected_c", "expected_eps"),
    [
        (
            {
                "grad_modif_type": "std_norm",
                "grad_modif_params": ["c=0.42", "eps=1e-8", "layer_type=v1"],
            },
            GradientSTDNormLayerV1,
            0.42,
            1e-8,
        ),
        (
            {
                "grad_modif_type": "std_norm",
                "grad_modif_params": ["c=0", "eps=1e-5", "layer_type=v2"],
            },
            GradientSTDNormLayerV2,
            0.0,
            1e-5,
        ),
        (
            {
                "grad_modif_type": "std_norm",
                "grad_modif_params": ["c=1", "eps=1.1", "layer_type=v3"],
            },
            GradientSTDNormLayerV3,
            1.0,
            1.1,
        ),
        (
            {
                "grad_modif_type": "std_norm",
                "grad_modif_params": ["c=-0.1", "eps=0.0", "layer_type=v1"],
            },
            GradientSTDNormLayerV1,
            -0.1,
            0.0,
        ),
    ],
    indirect=["args_mock"],
)
def test_get_grad_modif_fn_std_norm(args_mock, expected_layer_type, expected_c, expected_eps):
    std_norm = get_grad_modif_fn(args_mock)()
    assert isinstance(std_norm, expected_layer_type)
    assert std_norm.c == expected_c
    assert std_norm.eps == expected_eps


@pytest.mark.parametrize(
    "args_mock",
    [
        {
            "grad_modif_type": "std_norm",
        },
        {"grad_modif_type": "std_norm", "grad_modif_params": ["cc=1"]},
        {
            "grad_modif_type": "std_norm",
            "grad_modif_params": ["c=0.42", "eps=1e-8", "layer_type=v0"],
        },
        {
            "grad_modif_type": "std_norm",
            "grad_modif_params": ["c=0.42", "eps=1e-8", "layer_type=v4"],
        },
        {
            "grad_modif_type": "scale_norm",
        },
        {"grad_modif_type": "scale_norm", "grad_modif_params": ["k=1"]},
        {
            "grad_modif_type": "scale_norm",
            "grad_modif_params": ["k='auto'", "eps=1e-8"],
        },
        {
            "grad_modif_type": "scale_norm",
            "grad_modif_params": ["eps=1e-8"],
        },
        {
            "grad_modif_type": "activation_norm",
            "grad_modif_params": ["eps=1e-8", "norm_dims=()"],
        },
        {
            "grad_modif_type": "activation_norm",
            "grad_modif_params": ["eps=1e-8", "norm_dims=(0, 1, 3)"],
        },
    ],
    indirect=True,
)
def test_get_grad_modif_fn_error_handling(args_mock):
    with pytest.raises(ValueError):
        get_grad_modif_fn(args_mock)


@pytest.mark.parametrize(
    "args_mock",
    [{"grad_modif_params": ["c=1"], "grad_modif_type": None}, {"grad_modif_type": None}],
    indirect=True,
)
def test_get_grad_modif_fn_returns_none(args_mock):
    assert get_grad_modif_fn(args_mock) is None


@pytest.mark.parametrize(
    ("args_mock", "expected_layer_type", "expected_k", "expected_eps", "expected_c", "expected_norm_dims"),
    [
        (
            {
                "grad_modif_type": "scale_norm",
                "grad_modif_params": ["k=auto", "eps=1e-8", "c=1", "norm_dims=(0,1,2)"],
            },
            GradientScaleNormLayer,
            "auto",
            1e-8,
            1,
            (0, 1, 2),
        ),
        (
            {
                "grad_modif_type": "scale_norm",
                "grad_modif_params": ["k=0.5", "eps=1e-5", "c=0.5", "norm_dims=(1,2)"],
            },
            GradientScaleNormLayer,
            0.5,
            1e-5,
            0.5,
            (1, 2),
        ),
        (
            {
                "grad_modif_type": "scale_norm",
                "grad_modif_params": ["k=1", "eps=0", "c=0", "norm_dims=(2,)"],
            },
            GradientScaleNormLayer,
            1.0,
            0,
            0,
            (2,),
        ),
    ],
    indirect=["args_mock"],
)
def test_get_grad_modif_fn_scale_norm(
    args_mock, expected_layer_type, expected_k, expected_eps, expected_c, expected_norm_dims
):
    scale_norm_layer = get_grad_modif_fn(args_mock)()
    assert isinstance(scale_norm_layer, expected_layer_type)
    assert scale_norm_layer.k == expected_k
    assert scale_norm_layer.eps == expected_eps
    assert scale_norm_layer.c == expected_c
    assert scale_norm_layer.norm_dims == expected_norm_dims


@pytest.mark.parametrize(
    ("args_mock", "expected_layer_type", "expected_norm_dims", "expected_eps"),
    [
        (
            {
                "grad_modif_type": "activation_norm",
                "grad_modif_params": ["eps=1e-8", "norm_dims=(0, 1, 2)"],
            },
            GradientActivationNormLayer,
            (0, 1, 2),
            1e-8,
        ),
        (
            {
                "grad_modif_type": "activation_norm",
                "grad_modif_params": ["eps=1e-5", "norm_dims=[0, 1]"],
            },
            GradientActivationNormLayer,
            (0, 1),
            1e-5,
        ),
        (
            {
                "grad_modif_type": "activation_norm",
                "grad_modif_params": ["norm_dims=(2,)", "eps=0"],
            },
            GradientActivationNormLayer,
            (2,),
            0,
        ),
    ],
    indirect=["args_mock"],
)
def test_get_grad_modif_fn_activation_norm(args_mock, expected_layer_type, expected_norm_dims, expected_eps):
    scale_norm_layer = get_grad_modif_fn(args_mock)()
    assert isinstance(scale_norm_layer, expected_layer_type)
    assert scale_norm_layer.eps == expected_eps
    assert scale_norm_layer.norm_dims == expected_norm_dims
