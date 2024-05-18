from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LayerGradModifPlacement:
    post_layer: bool = False
    post_norm: bool = False
    post_add: bool = False

    @staticmethod
    def full_mod() -> "LayerGradModifPlacement":
        return LayerGradModifPlacement(True, True, True)
    
    def is_empty(self) -> bool:
        return not (self.post_norm or self.post_norm or self.post_add)

    @staticmethod
    def all_placements() -> Iterable["LayerGradModifPlacement"]:
        for i in range(8):
            yield LayerGradModifPlacement(
                (i >> 0) & 1 == 1,
                (i >> 1) & 1 == 1,
                (i >> 2) & 1 == 1
            )


@dataclass(frozen=True)
class BlockGradModifPlacement:
    attn_mod: LayerGradModifPlacement = LayerGradModifPlacement()
    ff_mod: LayerGradModifPlacement = LayerGradModifPlacement()

    @staticmethod
    def create_full() -> "BlockGradModifPlacement":
        return BlockGradModifPlacement(
            LayerGradModifPlacement.full_mod(),
            LayerGradModifPlacement.full_mod()
        )

    @staticmethod
    def all_placements() -> Iterable["BlockGradModifPlacement"]:
        for attn in LayerGradModifPlacement.all_placements():
            for ff in LayerGradModifPlacement.all_placements():
                yield BlockGradModifPlacement(attn, ff)
    


'''
@dataclass(frozen=True)
class GradModifPlacement:

    ModifPlacement = Literal[
        "post_attn",
        "post_attn_norm",
        "post_attn_add",
        "post_ff",
        "post_ff_norm",
        "post_ff_add"
        ]

    post_attn: bool = False
    post_attn_norm: bool = False
    post_attn_add: bool = False
    post_ff: bool = False
    post_ff_norm: bool = False
    post_ff_add: bool = False

    def is_post_layer(self) -> bool:
        return self.post_attn or self.post_ff
    
    def is_post_norm(self) -> bool:
        return self.post_attn_norm or self.post_ff_norm
    
    def is_post_add(self) -> bool:
        return self.post_attn_add or self.post_ff_add

    def affects_attention(self) -> bool:
        return self.post_attn or self.post_attn_norm or self.post_attn_add

    def affects_ff(self) -> bool:
        return self.post_ff or self.post_ff_norm or self.post_ff_add

    def to_list(self) -> list[ModifPlacement]:
        return [field.name for field in fields(self) if getattr(self, field.name) is True]
    
    @staticmethod
    def from_list(vals: list[ModifPlacement]) -> "GradModifPlacement":
        return GradModifPlacement(**{v: True for v in vals})
    
    @staticmethod
    def max_placement() -> "GradModifPlacement":
        return GradModifPlacement(post_attn=True, post_attn_norm=True, post_attn_add=True, post_ff=True, post_ff_norm=True, post_ff_add=True)
    
    def _powerset(self, iterable):
        # https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

    def superset(self) -> list["GradModifPlacement"]:
        return [GradModifPlacement.from_list(s) for s in self._powerset(self.to_list())]
'''