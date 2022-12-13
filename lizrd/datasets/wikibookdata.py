import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import BertTokenizer


class ProcessedExample(object):
    def __init__(self, sentence, processor):
        self.tokens = processor.tokenize_text(sentence)
        self.tokens = processor.pad_tokens(self.tokens)
        special_token_mask = processor.special_token_mask(self.tokens)
        self.mask_mask = processor.get_mask_mask(special_token_mask)
        self.masked_tokens = processor.mask_tokens(self.tokens, self.mask_mask)


class ProcessedBatch(object):
    def __init__(self, processed_examples):
        self.tokens = self._make_tensor(
            [example.tokens for example in processed_examples]
        )
        self.mask_mask = self._make_tensor(
            [example.mask_mask for example in processed_examples]
        )
        self.masked_tokens = self._make_tensor(
            [example.masked_tokens for example in processed_examples]
        )

        assert self.tokens.shape == self.masked_tokens.shape
        assert self.tokens.shape == self.mask_mask.shape

    def _make_tensor(self, list_of_token_lists):
        matrix = np.array(list_of_token_lists)
        return torch.from_numpy(matrix)

    def to_(self, device):
        self.tokens = self.tokens.to(device)
        self.masked_tokens = self.masked_tokens.to(device)
        self.mask_mask = self.mask_mask.to(device)
        return self


class SentencePairProcessor(object):
    def __init__(self, max_total_length=128, mask_percent=0.15, rng=None):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_total_length = max_total_length
        self.max_sentence_length = (max_total_length - 4) // 2
        self.mask_token = "[MASK]"
        self.sep_token = "[SEP]"
        self.cls_token = "[CLS]"
        self.pad_token = "[PAD]"
        self.mask_id = self.tokenizer._convert_token_to_id("[MASK]")
        self.cls_id = self.tokenizer._convert_token_to_id("[CLS]")
        self.sep_id = self.tokenizer._convert_token_to_id("[SEP]")
        self.pad_id = self.tokenizer._convert_token_to_id("[PAD]")
        self.special_tokens = [
            self.cls_token,
            self.sep_token,
            self.pad_token,
            self.mask_token,
        ]
        self.special_token_ids = [self.cls_id, self.sep_id, self.pad_id, self.mask_id]
        self.mask_percent = mask_percent
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def process(self, sentence):
        return ProcessedExample(sentence, self)

    def tokenize_text(self, sentence_text):
        # note: tokenizer.encode _claims_ to be equivalent. This isn't true.
        return self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(sentence_text)
        )

    def special_token_mask(self, sentence_tokens):
        return np.isin(sentence_tokens, self.special_token_ids)

    def get_mask_mask(self, special_token_mask):
        mask_mask = self.rng.binomial(1, self.mask_percent, len(special_token_mask))
        mask_mask = mask_mask.astype(bool)
        mask_mask = np.where(special_token_mask, 0, mask_mask)
        return mask_mask

    def mask_tokens(self, sentence_tokens, mask_mask):
        sentence_tokens = np.where(mask_mask, self.mask_id, sentence_tokens)
        return sentence_tokens

    def pad_tokens(self, sentence_tokens):
        if len(sentence_tokens) > self.max_total_length:
            sentence_tokens = sentence_tokens[: self.max_total_length]
        return sentence_tokens + [self.pad_id] * (
            self.max_total_length - len(sentence_tokens)
        )


def process_wiki_text(document_text, chunk_length: int = 450):
    "splits document into a list of chunks of specified length"
    chunks = [
        document_text[i : i + chunk_length]
        for i in range(0, len(document_text), chunk_length)
    ]
    return chunks


def process_book_text(document_sentences, chunk_length: int = 450):
    """
    glue together sentences into chunks of at least `chunk_length`
    :param document_sentences: list of strings, each string is a sentence
    :return: list of strings, each string is a chunk of length at least 450
    """
    chunks = []
    current_chunk = ""
    for sentence in document_sentences:
        if len(current_chunk) + len(sentence) > chunk_length:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += sentence
    return chunks


class WikiBookDataset:
    def __init__(self, rng=random):
        self.examples_buffer = []
        self.dataset_wiki = load_dataset("wikipedia", "20220301.en")["train"]
        self.dataset_book = load_dataset("bookcorpus")["train"]
        self.rng = rng

        self.buffer_refill_to = 10000
        self.buffer_refill_from = 0
        self.min_sentence_length = 40
        self.bookcorpus_chance = 0.5
        self.bookcorpus_lines = len(self.dataset_book) // len(self.dataset_wiki) + 1
        self.bookcorpus_chance = self.bookcorpus_chance / 100 * self.bookcorpus_lines
        self.bookcorpus_lines = 100  # the above is very approximate
        self.wikipedia_chance = 1.0 - self.bookcorpus_chance
        print("bookcorpus_lines:", self.bookcorpus_lines)
        print("bookcorpus_chance:", self.bookcorpus_chance)

    def get_example(self):
        if len(self.examples_buffer) <= self.buffer_refill_from:
            self._refill_buffer()
        example = self.examples_buffer.pop()
        return example

    def get_batch(self, batch_size):
        batch = [self.get_example() for _ in range(batch_size)]
        return batch

    def _refill_buffer(self):
        while len(self.examples_buffer) <= self.buffer_refill_to:
            self._add_examples(self._get_random_document())
        self.rng.shuffle(self.examples_buffer)

    def _get_random_document(self):
        if self.rng.random() < self.wikipedia_chance:
            document_text = self.dataset_wiki[
                self.rng.randint(0, len(self.dataset_wiki) - 1)
            ]["text"]
            documents_sentences = process_wiki_text(document_text)
            assert isinstance(documents_sentences, list)
            assert isinstance(documents_sentences[0], str)
        else:
            linebegin = self.rng.randint(
                0, len(self.dataset_wiki) - 1 - self.bookcorpus_lines
            )
            lineend = linebegin + self.bookcorpus_lines
            documents_sentences = self.dataset_book[linebegin:lineend]["text"]
            documents_sentences = process_book_text(documents_sentences)
            assert isinstance(documents_sentences, list)
            assert isinstance(documents_sentences[0], str)
        return documents_sentences

    def _add_examples(self, param):
        """This version simply filters out all sentences that are too short, then adds all remaining sentences to the buffer."""

        document_sentences = [
            sentence for sentence in param if len(sentence) > self.min_sentence_length
        ]
        self.examples_buffer += document_sentences


class ProcessedDataset:
    def __init__(self, dataset, processor):
        assert isinstance(dataset, WikiBookDataset)
        self.dataset = dataset
        assert isinstance(processor, SentencePairProcessor)
        self.processor = processor

    def get_example(self):
        example = self.dataset.get_example()
        processed_example = self.processor.process(example)
        return processed_example


class ParallelCompatibleDataset(IterableDataset):
    def __init__(self, dataset: ProcessedDataset, batch_size: int, seed: int = 42):
        super().__init__()
        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            seed = self.seed
        else:
            seed = self.seed + worker_info.id
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.dataset.dataset.rng = self.rng
        self.dataset.processor.rng = self.np_rng
        while True:
            yield self.dataset.get_example()


class ProcessedDatasetWrapper:
    """
    This class is a wrapper around a ProcessedDataset that provides a get_batch() method that returns a batch of processed examples.
    Takes care of seeding the rng, collating the examples into a batch, and moving the batch to the correct device.
    Allows multiple workers to be used.
    To make `get_batch` return the same sequence of batches, keep the seed, batch_size and num_workers unchanged.
    """

    def _collate_fn(self, batch) -> ProcessedBatch:
        return ProcessedBatch(batch)

    def __init__(
        self,
        pdataset: ProcessedDataset,
        device: torch.device,
        batch_size: int,
        num_workers: int = 8,
        seed: int = 42,
    ):
        self.pdataset = pdataset
        self.device = device
        self.dataloader = iter(
            DataLoader(
                ParallelCompatibleDataset(pdataset, batch_size=batch_size, seed=seed),
                num_workers=num_workers,
                batch_size=batch_size,
                collate_fn=self._collate_fn,
                shuffle=False,
            )
        )

    def get_batch(self) -> ProcessedBatch:
        return next(self.dataloader).to_(self.device)
