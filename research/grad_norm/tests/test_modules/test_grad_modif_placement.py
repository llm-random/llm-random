import pytest

from research.grad_norm.modules.grad_modif_placement import (
    BlockGradModifPlacement,
    LayerGradModifPlacement,
)


def test_layer_all_placements():
    all_placements = set(LayerGradModifPlacement.all_placements())

    assert len(all_placements) == 2**3


def test_blok_all_placements():
    all_placements = set(BlockGradModifPlacement.all_placements())

    assert len(all_placements) == 2**6


def test_create_from_list():
    placements = ["post_attn", "post_ff_norm", "post_ff_add"]
    placement = BlockGradModifPlacement.from_list(placements)

    assert placement == BlockGradModifPlacement(
        attn_mod=LayerGradModifPlacement(post_layer=True),
        ff_mod=LayerGradModifPlacement(post_norm=True, post_add=True),
    )


def test_create_from_list_error_handling():
    placements = ["post_attn", "post_ff_norm", "post_ff_add", "post_atn_add"]

    with pytest.raises(ValueError):
        BlockGradModifPlacement.from_list(placements)
