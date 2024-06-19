import torch
import torch.nn as nn

from lizrd.core.misc import Linear


def choose_indeces_to_reduce(batch_size, seq_len, result_seq_len, n_tokens_to_reduce):
    """
    This function generates random indices to keep and indices of tokens to reduce.

    - batch_size: the number of sequences in the batch
    - seq_len: the length of the sequences in the batch from dataloader
    - result_seq_len: the length of the sequences after token reduction
    - n_tokens_to_reduce: the number of tokens to reduce

    If n_tokens_to_reduce is less than the difference between the original sequence length and the result sequence length,
    remaining tokens at the end of the sequence will be dropped upfront e.g.
    seq_len = 7, result_seq_len = 3, n_tokens_to_reduce = 2
    We start with ids:
    0 1 2 3 4 5 6
    We drop the last two tokens:
    0 1 2 3 4

    With the remaining tokens, we randomly permute them and split them into two groups, which are later returned:
    - indices_to_keep: tensor of shape (batch_size * result_seq_len) containing the indices of the tokens to keep
    - indices_to_reduce: tensor of shape (batch_size * (seq_len - result_seq_len)) containing the indices of the tokens to reduce
    """
    assert result_seq_len + n_tokens_to_reduce <= seq_len, "Too many tokens to reduce"

    # We do not want to reduce the last token, because we do not have a token to merge it with or it does not make sense to drop the last token
    is_last_undroppable = result_seq_len + n_tokens_to_reduce == seq_len

    permutation_len = result_seq_len + n_tokens_to_reduce
    if is_last_undroppable:
        permutation_len -= 1

    random_perms = [torch.randperm(permutation_len) for _ in range(batch_size)]
    if is_last_undroppable:
        pairs = [
            (
                torch.cat(
                    (
                        torch.sort(permutation[: result_seq_len - 1])[0],
                        torch.tensor(
                            [permutation_len]
                        ),  # Add the last token, so it is not reduced so we can merge antecendent token
                    )
                ),
                permutation[result_seq_len - 1 :],
            )
            for permutation in random_perms
        ]
    else:
        pairs = [
            (
                torch.sort(permutation[:result_seq_len])[0],
                permutation[result_seq_len:],
            )
            for permutation in random_perms
        ]

    for i, (indices_to_keep, indices_to_reduce) in enumerate(pairs):
        indices_to_keep += i * seq_len
        indices_to_reduce += i * seq_len

    indices_to_keep, indices_to_reduce = zip(*pairs)

    indices_to_keep, indices_to_reduce = torch.stack(indices_to_keep), torch.stack(
        indices_to_reduce
    )
    indices_to_keep = torch.flatten(indices_to_keep)
    indices_to_reduce = torch.flatten(indices_to_reduce)
    return indices_to_keep, indices_to_reduce


class TokenDroppingLayerOld(nn.Module):
    """
    This function randomly selects a `result_seq_len` subset of tokens from the input
    """

    def __init__(self, result_seq_len):
        super(TokenDroppingLayer, self).__init__()
        self.result_seq_len = result_seq_len

    def _random_indeces(self, batch_size, seq_len):
        indices_to_keep = torch.stack(
            [
                torch.sort(torch.randperm(seq_len)[: self.result_seq_len])[0]
                for _ in range(batch_size)
            ]
        )
        return indices_to_keep

    def _batched_index_select(self, input, dim, index):
        """
        origin: https://discuss.pytorch.org/t/batched-index-select/9115/8
        input: B x * x ... x *
        dim: 0 < scalar
        index: B x M
        """
        views = [input.shape[0]] + [
            1 if i != dim else -1 for i in range(1, len(input.shape))
        ]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        assert self.result_seq_len <= seq_len
        indices_to_keep = self._random_indeces(batch_size, seq_len).to(x.device)
        self.indices_to_keep = indices_to_keep
        return self._batched_index_select(x, 1, indices_to_keep)


class TokenDroppingLayer(nn.Module):
    """
    This function randomly selects a `result_seq_len` subset of tokens from the input
    """

    def __init__(self, result_seq_len):
        super(TokenDroppingLayer, self).__init__()
        self.result_seq_len = result_seq_len

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        assert self.result_seq_len <= seq_len

        indices_to_keep, _ = choose_indeces_to_reduce(
            batch_size, seq_len, self.result_seq_len, seq_len - self.result_seq_len
        )
        self.indices_to_keep = indices_to_keep
        selected_tokens = x.view(batch_size * seq_len, -1).index_select(
            0, indices_to_keep.to(x.device)
        )
        return selected_tokens.view(batch_size, self.result_seq_len, -1)


class TokenMergingLayer(nn.Module):
    """
    This function randomly selects a `result_seq_len` subset of tokens from the input
    """

    def __init__(self, result_seq_len, dm, init_type="kaiming_uniform", init_scale=1.0):
        super(TokenMergingLayer, self).__init__()
        self.result_seq_len = result_seq_len

        self.merge_linear_projection = Linear(
            dm, dm, init_type=init_type, init_scale=init_scale, bias=False
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        assert self.result_seq_len <= seq_len

        indices_to_keep, indices_to_reduce = choose_indeces_to_reduce(
            batch_size, seq_len, self.result_seq_len, seq_len - self.result_seq_len
        )
        indices_to_keep, indices_to_reduce = indices_to_keep.to(
            x.device
        ), indices_to_reduce.to(x.device)

        x = x.view(-1, x.shape[-1])
        reduced_tokens = torch.index_select(x, 0, indices_to_reduce)
        transformed_reduced_tokens = self.merge_linear_projection(
            reduced_tokens
        ).float()

        x.index_add_(0, indices_to_reduce + 1, transformed_reduced_tokens)
        kept_tokens = torch.index_select(x, 0, indices_to_keep)

        self.indices_to_keep, self.indices_to_reduce = indices_to_keep, indices_to_reduce
        return kept_tokens.view(batch_size, self.result_seq_len, -1)

class TokenMergingLayerOld(nn.Module):
    """
    This function randomly selects a `result_seq_len` subset of tokens from the input
    """

    def __init__(
        self, result_seq_len, dm=None, init_type="kaiming_uniform", init_scale=1.0
    ):
        super(TokenMergingLayer, self).__init__()
        self.result_seq_len = result_seq_len

        self.transform_to_merge = Linear(
            dm, dm, init_type=init_type, init_scale=init_scale, bias=False
        )

    def _random_indeces2(self, batch_size, seq_len):

        random_perms = [torch.randperm(seq_len - 1) for _ in range(batch_size)]

        pairs = [
            (
                torch.cat(
                    (
                        torch.sort(permutation[: self.result_seq_len - 1])[0],
                        torch.tensor([seq_len - 1]),
                    )
                ),
                permutation[self.result_seq_len - 1 :],
            )
            for permutation in random_perms
        ]

        for i, (indices_to_keep, indices_to_reduct) in enumerate(pairs):
            indices_to_keep += i * seq_len
            indices_to_reduct += i * seq_len

        indices_to_keep, indices_to_reduct = zip(*pairs)

        indices_to_keep, indices_to_reduct = torch.stack(indices_to_keep), torch.stack(
            indices_to_reduct
        )
        indices_to_keep = torch.flatten(indices_to_keep)
        indices_to_reduct = torch.flatten(indices_to_reduct)
        return indices_to_keep, indices_to_reduct

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        assert self.result_seq_len <= seq_len
        x = x.view(-1, x.shape[-1])

        indices_to_keep, indices_to_reduce = self._random_indeces2(batch_size, seq_len)
        indices_to_keep, indices_to_reduce = indices_to_keep.to(
            x.device
        ), indices_to_reduce.to(x.device)
        self.indices_to_reduce = indices_to_reduce
        self.indices_to_keep = indices_to_keep

        reduced_tokens = torch.index_select(x, 0, indices_to_reduce)
        transformed_reduced_tokens = self.transform_to_merge(reduced_tokens).float()

        x.index_add_(0, indices_to_reduce + 1, transformed_reduced_tokens)
        kept_tokens = torch.index_select(x, 0, indices_to_keep)

        return kept_tokens.view(batch_size, self.result_seq_len, -1)


class TokenReductionEmbedding(nn.Module):
    def __init__(self, base_embedding_layer, reduction_layer):
        super().__init__()
        self.base_embedding_layer = base_embedding_layer
        self.reduction_layer = reduction_layer

    def forward(self, x):
        x = self.base_embedding_layer(x)
        if self.training:
            x = self.reduction_layer(x)
        return x
