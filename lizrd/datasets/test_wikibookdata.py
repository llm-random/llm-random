from lizrd.datasets import wikibookdata
import lizrd.datasets.processed_batch
from lizrd.datasets.wikibookdata import get_processed_dataset
from lizrd.support.test_utils import GeneralTestCase, skip_test, heavy_test
import pickle


class TestWikibookdata(GeneralTestCase):
    @skip_test(reason="Deprecated, implementation changed")
    def test_integration_deprecated(self):
        max_len = 100
        batch_size = 32
        raw_dataset = wikibookdata.WikiBookDataset()
        processor = wikibookdata.SentencePairProcessor(max_total_length=max_len)
        dataset = wikibookdata.ProcessedDataset(raw_dataset, processor)
        processed_batch = dataset.get_batch(batch_size)
        isinstance(processed_batch, lizrd.datasets.processed_batch.ProcessedBatch)
        self.assertShape(processed_batch.swapped, (batch_size,))
        self.assertEqual(processed_batch.swapped.sum(), batch_size // 2)
        self.assertShape(processed_batch.masked_tokens, (batch_size, max_len))
        self.assertShape(processed_batch.tokens, (batch_size, max_len))
        self.assertShape(processed_batch.special_token_mask, (batch_size, max_len))
        self.assertShape(processed_batch.special_token_mask, (batch_size, max_len))

    @heavy_test
    def test_consistency(self):
        ds = get_processed_dataset(32, 128, 0.15, "cpu", 2, 1, use_dummy_dataset=True)
        batch = ds.get_batch()
        # compare batch with saved batch
        with open("lizrd/datasets/test_batch.pkl", "rb") as f:
            saved_batch = pickle.load(f)
        self.assertTensorEqual(batch.masked_tokens, saved_batch.masked_tokens)
        self.assertTensorEqual(batch.tokens, saved_batch.tokens)
        self.assertTensorEqual(batch.mask_mask, saved_batch.mask_mask)

    @heavy_test
    def test_integration(self):
        bs = 32
        max_len = 128
        ds = get_processed_dataset(
            bs, max_len, 0.15, "cpu", 2, 1, use_dummy_dataset=True
        )
        batch = ds.get_batch()
        self.assertShape(batch.masked_tokens, (bs, max_len))
        self.assertShape(batch.tokens, (bs, max_len))
        self.assertShape(batch.mask_mask, (bs, max_len))
