from torch.utils.data import Dataset, DataLoader
import torch

from lizrd.support.test_utils import GeneralTestCase
from lizrd.text.data import LLMBatch, LLMExample
from research.datasets import DataloaderWrapper


class TestDataLoaderWrapper(GeneralTestCase):
    def test_basic(self):
        class MockDataset(Dataset):
            def __init__(self, size=1000):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, index):
                return LLMExample(
                    input_ids=[index] * 3,
                    target_ids=[index] * 3,
                    should_calculate_loss=[1] * 3,
                )

        dataset = MockDataset()
        dataloader = DataLoader(
            dataset, batch_size=8, collate_fn=LLMBatch, shuffle=False
        )
        wrapper = DataloaderWrapper(dataloader, torch.device("cpu"))

        b = wrapper.get_batch(2)
        expected = torch.Tensor([[0, 0, 0], [1, 1, 1]])
        self.assertTensorEqual(b.input_ids, expected)
        self.assertTensorEqual(b.target_ids, expected)

        b = wrapper.get_batch(2)
        expected = torch.Tensor([[2, 2, 2], [3, 3, 3]])
        self.assertTensorEqual(b.input_ids, expected)
        self.assertTensorEqual(b.target_ids, expected)

        b = wrapper.get_batch(2)
        expected = torch.Tensor([[4, 4, 4], [5, 5, 5]])
        self.assertTensorEqual(b.input_ids, expected)
        self.assertTensorEqual(b.target_ids, expected)

        b = wrapper.get_batch(4)
        expected = torch.Tensor([[8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11]])
        self.assertTensorEqual(b.input_ids, expected)
        self.assertTensorEqual(b.target_ids, expected)

        b = wrapper.get_batch(4)
        expected = torch.Tensor(
            [[12, 12, 12], [13, 13, 13], [14, 14, 14], [15, 15, 15]]
        )
        self.assertTensorEqual(b.input_ids, expected)
        self.assertTensorEqual(b.target_ids, expected)

        b = wrapper.get_batch(8)
        expected = torch.Tensor(
            [
                [16, 16, 16],
                [17, 17, 17],
                [18, 18, 18],
                [19, 19, 19],
                [20, 20, 20],
                [21, 21, 21],
                [22, 22, 22],
                [23, 23, 23],
            ]
        )
        self.assertTensorEqual(b.input_ids, expected)
        self.assertTensorEqual(b.target_ids, expected)
