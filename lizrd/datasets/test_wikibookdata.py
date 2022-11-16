from lizrd.datasets import wikibookdata
from lizrd.support.test_utils import GeneralTestCase


class TestWikibookdata(GeneralTestCase):
    def test_integration(self):
        max_len = 100
        batch_size = 32
        raw_dataset = wikibookdata.WikiBookDataset()
        processor = wikibookdata.SentencePairProcessor(max_total_length=max_len)
        dataset = wikibookdata.ProcessedDataset(raw_dataset, processor)
        processed_batch = dataset.get_batch(batch_size)
        isinstance(processed_batch, wikibookdata.ProcessedBatch)
        self.assertShape(processed_batch.swapped, (batch_size,))
        self.assertEqual(processed_batch.swapped.sum(), batch_size // 2)
        self.assertShape(processed_batch.masked_tokens, (batch_size, max_len))
        self.assertShape(processed_batch.tokens, (batch_size, max_len))
        self.assertShape(processed_batch.special_token_mask, (batch_size, max_len))
        self.assertShape(processed_batch.special_token_mask, (batch_size, max_len))
