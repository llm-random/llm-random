# %%

import torch

# from einops import einops
# from sum
N = 10
DM = 5
HEAD_DIM = 7
NE = 2
x = torch.rand(N, DM)
router = torch.nn.Linear(DM, NE)
experts_k = torch.randn(NE, DM, HEAD_DIM)
experts_v = torch.randn(NE, DM, HEAD_DIM)
gating = router(x).softmax(-1)
topk = gating.topk(k=1, dim=-1)
topk_values = topk.values
topk_indices = topk.indices
gating_top1 = torch.zeros_like(gating)
indicator = torch.zeros_like(gating)
indicator.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_values))
C = N // NE
tokens_per_expert = indicator.sum(dim=0).type(torch.int64)
print(tokens_per_expert)
token_idx_within_expert = (indicator.cumsum(dim=0) * indicator).type(torch.int64)
token_is_dropped = (token_idx_within_expert > C).sum(1, keepdim=True)
truncated_token_idx_within_expert = token_idx_within_expert * (1 - token_is_dropped)
print(token_idx_within_expert)
# experts_input = x.reshape(B * T, -1)
experts_input = torch.zeros(NE, C + 1, DM)
# experts_input.
print(truncated_token_idx_within_expert.shape)
print(experts_input.shape)
print(x.shape)
print(truncated_token_idx_within_expert.T.unsqueeze(-1).shape)
experts_input.scatter_(
    dim=1,
    index=truncated_token_idx_within_expert.T.unsqueeze(-1).expand(NE, -1, DM),
    src=x.unsqueeze(0).expand(NE, -1, -1),
)
print(experts_input.shape)
experts_input.unsqueeze_(-2)
print(experts_input.shape)
experts_k.unsqueeze_(1)
print(experts_k.shape)
k = torch.matmul(experts_input, experts_k).squeeze(-2)
k = k.reshape(-1, k.shape[-1])[
    (
        (truncated_token_idx_within_expert + torch.arange(0, NE) * (C + 1))
        * (truncated_token_idx_within_expert != 0)
    ).sum(-1)
]
k.shape
# k[token_is_dropped] = torch.randn([1, HEAD_DIM])
# k.shape
# k.reshape(HEAD_DIM)

# print(k.shape)
# print(truncated_token_idx_within_expert.T.unsqueeze(-1).expand(NE, -1, HEAD_DIM).shape)
# k_gathered = k.gather(
#     dim=1,
#     index=truncated_token_idx_within_expert.T.unsqueeze(-1).expand(NE, -1, HEAD_DIM),
# )
# # k_gat
# print(k_gathered.shape)
# print(k_gathered.shape)

# %%
k_gathered
# experts_output =

# print(experts_input.shape)


# is_within_capacity = token_idx_within_expert.reshape(B, T, self.n_head) <= capacity
# tru
# gating = einops.eins(
#     x, router, "a b, b c -> a c"
# )
# print(gating)
# %%
