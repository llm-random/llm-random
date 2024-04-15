import torch
import torch.nn as nn


class TokenReductionLayer(nn.Module):
    """
    This function randomly selects a `result_seq_len` subset of tokens from the input
    """

    def __init__(self, result_seq_len):
        super(TokenReductionLayer, self).__init__()
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


class TokenReductionLLM(nn.Module):

    def __init__(self, embedding_layer, encoder_tower, head, reduced_number_of_tokens):
        super(TokenReductionLLM, self).__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder_tower
        self.head = head
        self.reduced_number_of_tokens = reduced_number_of_tokens

    def forward(self, x):
        if self.training:
            x = self.embedding_layer(x)
        else:
            x = self.embedding_layer.normal(x)

        x = self.encoder(x)
        x = self.head(x)
        return x

    def get_chosen_indices(self):
        return self.embedding_layer.token_reduction.indices_to_keep
