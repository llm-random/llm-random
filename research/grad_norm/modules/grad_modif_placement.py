from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LayerGradModifPlacement:
    post_layer: bool = False
    post_norm: bool = False
    post_add: bool = False

    @staticmethod
    def create_full() -> "LayerGradModifPlacement":
        return LayerGradModifPlacement(True, True, True)

    def is_empty(self) -> bool:
        return not (self.post_norm or self.post_norm or self.post_add)

    @staticmethod
    def all_placements() -> Iterable["LayerGradModifPlacement"]:
        for i in range(8):
            yield LayerGradModifPlacement(
                (i >> 0) & 1 == 1, (i >> 1) & 1 == 1, (i >> 2) & 1 == 1
            )


@dataclass(frozen=True)
class BlockGradModifPlacement:
    attn_mod: LayerGradModifPlacement = LayerGradModifPlacement()
    ff_mod: LayerGradModifPlacement = LayerGradModifPlacement()

    @staticmethod
    def create_full() -> "BlockGradModifPlacement":
        return BlockGradModifPlacement(
            LayerGradModifPlacement.create_full(), LayerGradModifPlacement.create_full()
        )

    @staticmethod
    def all_placements() -> Iterable["BlockGradModifPlacement"]:
        for attn in LayerGradModifPlacement.all_placements():
            for ff in LayerGradModifPlacement.all_placements():
                yield BlockGradModifPlacement(attn, ff)
