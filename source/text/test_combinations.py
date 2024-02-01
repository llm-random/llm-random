from functools import partial

from source.support.test_utils import GeneralTestCase, heavy_test
from source.text.data import LLMBatch
from .datasets import WikiBookDataset
from .packers import GPTPacker
from .tokenizers import GPTTokenizer

from torch.utils.data import DataLoader

# TODO: Add more tests


class TestCombinations(GeneralTestCase):
    @heavy_test
    def test_integration(self):
        dataset = partial(WikiBookDataset, 1, True)
        tokenizer_maker = GPTTokenizer
        packer = GPTPacker(512, dataset, tokenizer_maker)

        num_workers = 2
        batch_size = 2
        iterations = 10
        dataloader = iter(
            DataLoader(
                packer,
                num_workers=num_workers,
                batch_size=batch_size,
                collate_fn=LLMBatch,
                shuffle=False,
            )
        )

        for i, batch in enumerate(dataloader):
            assert batch.input_ids.shape == (batch_size, 512)
            if i == iterations:
                break
