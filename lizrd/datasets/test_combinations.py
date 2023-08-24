from lizrd.datasets.processed_batch import GeneralBatch
from lizrd.support.test_utils import GeneralTestCase
from .datasets import WikiBookDataset, C4Dataset
from .packers import GPTPacker, BERTPacker
from .tokenizers import GPTTokenizer, BertTokenizer

from torch.utils.data import DataLoader


class TestCombinations(GeneralTestCase):
    def test_integration(self):
        dataset = WikiBookDataset(1, True)
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
                collate_fn=GeneralBatch,
                shuffle=False,
            )
        )

        for i, batch in enumerate(dataloader):
            assert len(batch.inp)
            if i == iterations:
                break
