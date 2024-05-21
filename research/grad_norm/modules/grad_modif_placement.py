from dataclasses import dataclass
from typing import Iterable, List, Literal


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
            yield LayerGradModifPlacement((i >> 0) & 1 == 1, (i >> 1) & 1 == 1, (i >> 2) & 1 == 1)


@dataclass(frozen=True)
class BlockGradModifPlacement:
    attn_mod: LayerGradModifPlacement = LayerGradModifPlacement()
    ff_mod: LayerGradModifPlacement = LayerGradModifPlacement()

    Placements = Literal["post_attn", "post_attn_norm", "post_attn_add", "post_ff", "post_ff_norm", "post_ff_add"]

    @staticmethod
    def create_full() -> "BlockGradModifPlacement":
        return BlockGradModifPlacement(LayerGradModifPlacement.create_full(), LayerGradModifPlacement.create_full())

    @staticmethod
    def all_placements() -> Iterable["BlockGradModifPlacement"]:
        for attn in LayerGradModifPlacement.all_placements():
            for ff in LayerGradModifPlacement.all_placements():
                yield BlockGradModifPlacement(attn, ff)

    @staticmethod
    def from_list(placemens: List[Placements]) -> "BlockGradModifPlacement":
        attn_init_kwargs = {}
        ff_init_wargs = {}
        for placement in placemens:
            if placement == "post_attn":
                attn_init_kwargs["post_layer"] = True
            elif placement == "post_attn_norm":
                attn_init_kwargs["post_norm"] = True
            elif placement == "post_attn_add":
                attn_init_kwargs["post_add"] = True
            elif placement == "post_ff":
                ff_init_wargs["post_layer"] = True
            elif placement == "post_ff_norm":
                ff_init_wargs["post_norm"] = True
            elif placement == "post_ff_add":
                ff_init_wargs["post_add"] = True
            else:
                raise ValueError(f"Unknown placement {placement}")

        return BlockGradModifPlacement(
            LayerGradModifPlacement(**attn_init_kwargs),
            LayerGradModifPlacement(**ff_init_wargs),
        )
