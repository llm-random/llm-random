from unittest.mock import MagicMock

import pytest

from research.grad_norm.build import get_grad_modif_fn, get_grad_modif_placement
from research.grad_norm.modules import BlockGradModifPlacement, GradientSTDNormLayer


def set_attrs_mock(m: MagicMock, d: dict):
    m.__getitem__.side_effect = d.__getitem__
    m.get.side_effect = d.get

    for k, v in d.items():
        if isinstance(v, dict):
            new_mock = MagicMock()
            set_attrs_mock(new_mock, v)
            setattr(m, k, new_mock)
        if isinstance(v, (list, str)):
            setattr(m, k, v)
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
    "args_mock",
    [
        {
            "grad_modif_type": "std_norm",
            "grad_modif_params": ["c=0.42"],
        },
    ],
    indirect=True,
)
def test_get_grad_modif_fn_std_norm(args_mock):
    std_norm = get_grad_modif_fn(args_mock)()
    assert isinstance(std_norm, GradientSTDNormLayer)
    assert std_norm.c == 0.42


@pytest.mark.parametrize(
    "args_mock",
    [
        {
            "grad_modif_type": "std_norm",
        },
        {"grad_modif_type": "std_norm", "grad_modif_params": ["cc=1"]},
    ],
    indirect=True,
)
def test_get_grad_modif_fn_error_handling(args_mock):
    with pytest.raises(ValueError):
        get_grad_modif_fn(args_mock)


@pytest.mark.parametrize(
    "args_mock",
    [{"grad_modif_params": ["c=1"]}, {}],
    indirect=True,
)
def test_get_grad_modif_fn_returns_none(args_mock):
    assert get_grad_modif_fn(args_mock) is None
