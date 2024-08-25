from unittest.mock import MagicMock

import pytest

from research.grad_norm.build import get_grad_modif_fn, get_grad_modif_placement
from research.grad_norm.modules import (
    BlockGradModifPlacement,
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
