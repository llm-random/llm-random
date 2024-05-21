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
