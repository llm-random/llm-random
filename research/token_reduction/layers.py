import torch
import torch.nn as nn

from lizrd.core.misc import Linear, LoggingLayer

GPT_EOT_ID = 50256


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


def make_available_ids(unembedded_input, result_seq_len, n_tokens_to_reduce, eot_id):
    _, train_seq_len = unembedded_input.shape
    assert (
        train_seq_len > result_seq_len + n_tokens_to_reduce
    ), "Not enough tokens in the input sequence"
    droppable_ids, saved_ids = [], []
    for row_number, tokens in enumerate(unembedded_input):
        droppable_tokens_mask = tokens[: result_seq_len + n_tokens_to_reduce] != eot_id
        droppable_ids.append(
            droppable_tokens_mask.nonzero(as_tuple=True)[0] + train_seq_len * row_number
        )
        saved_ids.append(
            (~droppable_tokens_mask).nonzero(as_tuple=True)[0]
            + train_seq_len * row_number
        )
    return droppable_ids, saved_ids


def choose_ids_to_reduce(available_ids, prior_saved_ids, n_tokens_to_reduce):
    permuted_ids = [ids[torch.randperm(len(ids))] for ids in available_ids]

    split_pairs = [
        (ids[:n_tokens_to_reduce], ids[n_tokens_to_reduce:]) for ids in permuted_ids
    ]

    sorted_results = [
        (torch.cat((save_ids, prior_ids)).sort()[0], reduce_ids)
        for ((reduce_ids, save_ids), prior_ids) in zip(split_pairs, prior_saved_ids)
    ]

    ids_to_save, ids_to_reduce = zip(*sorted_results)
    ids_to_save, ids_to_reduce = torch.stack(ids_to_save), torch.stack(ids_to_reduce)

    return ids_to_save, ids_to_reduce


def get_save_reduce_split(unembedded_input, result_seq_len, n_tokens_to_reduce, eot_id):
    available_ids, prior_saved_ids = make_available_ids(
        unembedded_input=unembedded_input,
        result_seq_len=result_seq_len,
        n_tokens_to_reduce=n_tokens_to_reduce,
        eot_id=eot_id,
    )

    ids_to_save, ids_to_reduce = choose_ids_to_reduce(
        available_ids, prior_saved_ids, n_tokens_to_reduce
    )

    return ids_to_save, ids_to_reduce


class TokenDroppingLayer(LoggingLayer):
    """
    This function randomly selects a `result_seq_len` subset of tokens from the input
    """

    def __init__(self, result_seq_len, scheduler=None):
        super(TokenDroppingLayer, self).__init__()
        self.result_seq_len = result_seq_len
        self.scheduler = scheduler

    def set_scheduler_step(self, step):
        if self.scheduler is not None:
            self.scheduler.set_step(step)

    def forward(self, token_reduction_input):
        x, unembedded_input = token_reduction_input
        batch_size, seq_len, _ = x.shape
        assert self.result_seq_len <= seq_len

        n_tokens_to_reduce = (
            seq_len - self.result_seq_len
            if self.scheduler is None
            else self.scheduler.value
        )
        self.update_cache_for_logging("reduced_tokens", n_tokens_to_reduce)

        indices_to_keep, _ = choose_indeces_to_reduce(
            batch_size, seq_len, self.result_seq_len, n_tokens_to_reduce
        )
        self.indices_to_keep = indices_to_keep
        selected_tokens = x.view(batch_size * seq_len, -1).index_select(
            0, indices_to_keep.to(x.device)
        )
        return selected_tokens.view(batch_size, self.result_seq_len, -1)

    def log_light(self):
        return {
            "reduced_tokens": self.logging_cache["reduced_tokens"],
        }


class TokenMergingLayer(LoggingLayer):
    """
    This function randomly selects a `result_seq_len` subset of tokens from the input
    """

    def __init__(
        self,
        result_seq_len,
        dm,
        scheduler=None,
        init_type="kaiming_uniform",
        init_scale=1.0,
    ):
        super(TokenMergingLayer, self).__init__()
        self.result_seq_len = result_seq_len
        self.scheduler = scheduler

        self.merge_linear_projection = Linear(
            dm, dm, init_type=init_type, init_scale=init_scale, bias=False
        )

    def set_scheduler_step(self, step):
        if self.scheduler is not None:
            self.scheduler.set_step(step)

    def forward(self, token_reduction_input):
        x, unembedded_input = token_reduction_input
        batch_size, seq_len, dm = x.shape
        assert self.result_seq_len <= seq_len

        n_tokens_to_reduce = (
            seq_len - self.result_seq_len
            if self.scheduler is None
            else self.scheduler.value
        )
        self.update_cache_for_logging("reduced_tokens", n_tokens_to_reduce)

        ids_to_save, ids_to_reduce = get_save_reduce_split(
            unembedded_input, self.result_seq_len, n_tokens_to_reduce, GPT_EOT_ID
        )

        ids_to_save, ids_to_reduce = ids_to_save.to(x.device), ids_to_reduce.to(
            x.device
        )

        x = x.view(-1, dm)
        reduced_tokens = torch.index_select(x, 0, ids_to_reduce)
        transformed_reduced_tokens = self.merge_linear_projection(
            reduced_tokens
        ).float()

        x.index_add_(0, ids_to_reduce + 1, transformed_reduced_tokens)
        kept_tokens = torch.index_select(x, 0, ids_to_save)

        self.indices_to_keep, self.indices_to_reduce = (
            ids_to_save,
            ids_to_reduce,
        )
        return kept_tokens.view(batch_size, self.result_seq_len, -1)

    def log_light(self):
        return {
            "reduced_tokens": self.logging_cache["reduced_tokens"],
        }


class TokenReductionEmbedding(nn.Module):
    def __init__(self, base_embedding_layer, reduction_layer):
        super().__init__()
        self.base_embedding_layer = base_embedding_layer
        self.reduction_layer = reduction_layer

    def set_scheduler_step(self, step):
        if self.reduction_layer is not None:
            self.reduction_layer.set_scheduler_step(step)

    def forward(self, x):
        unembedded_tokens = x.clone()
        x = self.base_embedding_layer(x)
        if self.training:
            x = self.reduction_layer((x, unembedded_tokens))
        return x
