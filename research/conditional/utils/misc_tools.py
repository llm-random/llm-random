import torch


def generate_shuffler_unshuffler(batch_size, seqlen, mix_whole_batch=False):
    if mix_whole_batch:
        shuffled_indices = torch.randperm(batch_size * seqlen)
    else:
        shuffled_indices = torch.randperm(seqlen)
    unshuffled_indices = shuffled_indices.argsort()

    if mix_whole_batch:

        def shuffle_tensor(tensor):
            reshape = tensor.reshape(-1, *tensor.shape[2:])
            shuffled_tensor = reshape[shuffled_indices]
            return shuffled_tensor.reshape(tensor.shape)

        def unshuffle_tensor(shuffled_tensor):
            reshape = shuffled_tensor.reshape(-1, *shuffled_tensor.shape[2:])
            return reshape[unshuffled_indices].reshape(shuffled_tensor.shape)

    else:

        def shuffle_tensor(tensor):
            return tensor[:, shuffled_indices, :]

        def unshuffle_tensor(shuffled_tensor):
            return shuffled_tensor[:, unshuffled_indices, :]

    return shuffle_tensor, unshuffle_tensor


def stable_softmax_temperature(x, temperature, dim=-1):
    x = x / temperature
    x = x - x.max(dim=dim, keepdim=True)[0]
    x = torch.exp(x)
    x = x / x.sum(dim=dim, keepdim=True)
    return x


def entropy(x, dim):
    ent = -torch.sum(x * torch.log(x + 1e-8), dim=dim)
    return ent
