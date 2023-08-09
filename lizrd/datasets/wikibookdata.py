import random
from typing import Literal

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import BertTokenizer, GPT2Tokenizer
from attr import define
from lizrd.datasets import wikibookdata
from neptune.utils import stringify_unsupported

from lizrd.datasets.processed_batch import (
    ProcessedBERTBatch,
    ProcessedBatch,
    ProcessedGPTBatch,
)
from lizrd.datasets.c4 import C4Dataset
from lizrd.support.logging import get_current_logger


class ProcessedBERTExample(object):
    def __init__(self, sentence, processor):
        self.tokens = processor.tokenize_text(sentence)
        self.tokens = processor.pad_tokens(self.tokens)
        special_token_mask = processor.special_token_mask(self.tokens)
        self.mask_mask = processor.get_mask_mask(special_token_mask)
        self.masked_tokens = processor.mask_tokens(self.tokens, self.mask_mask)


class ProcessedGPTExample(object):
    def __init__(self, sentence, processor):
        self.tokens = processor.tokenize_text(sentence)
        self.tokens, self.non_padded_mask = processor.pad_tokens(self.tokens)
        self.target_tokens = self.tokens[1:] + [processor.end_token_id]


@define
class MaskingReplacementConfig:
    replace_with_mask: float = 0.8
    replace_with_random: float = 0.1
    replace_with_original: float = 0.1

    def __attrs_post_init__(self):
        assert (
            self.replace_with_mask
            + self.replace_with_random
            + self.replace_with_original
        ) == 1.0


class GPTSentenceProcessor(object):
    def __init__(
        self,
        max_total_length=128,
    ):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.max_total_length = max_total_length
        end_token = "<|endoftext|>"
        self.end_token_id = self.tokenizer._convert_token_to_id(end_token)

    def process(self, sentence):
        return ProcessedGPTExample(sentence, self)

    def tokenize_text(self, sentence_text):
        # note: tokenizer.encode _claims_ to be equivalent. This isn't true.
        return self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(sentence_text)
        )

    def pad_tokens(self, sentence_tokens):
        if len(sentence_tokens) > self.max_total_length - 1:
            sentence_tokens = sentence_tokens[: self.max_total_length - 1]
        sentence_tokens.append(self.end_token_id)
        non_padding_length = len(sentence_tokens)
        padding_length = self.max_total_length - non_padding_length
        sentence_tokens = sentence_tokens + [self.end_token_id] * padding_length
        non_padded_mask = [1] * non_padding_length + [0] * padding_length
        return sentence_tokens, non_padded_mask


class BERTSentenceProcessor(object):
    def __init__(
        self,
        max_total_length=128,
        mask_percent=0.15,
        mask_replace_config=None,
        rng=None,
    ):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_total_length = max_total_length
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
        if mask_replace_config is None:
            mask_replace_config = MaskingReplacementConfig()
        self.mask_replace_config = mask_replace_config
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def process(self, sentence):
        return ProcessedBERTExample(sentence, self)

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

    def get_valid_random_tokens(self, tokens_count):
        # first 999 tokens are special tokens when using transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        special_tokens = 999
        return (
            self.rng.choice(self.tokenizer.vocab_size - special_tokens, tokens_count)
            + special_tokens
        )

    def mask_tokens(self, sentence_tokens, mask_mask):
        how_to_mask = self.rng.multinomial(
            1,
            [
                self.mask_replace_config.replace_with_mask,
                self.mask_replace_config.replace_with_random,
                self.mask_replace_config.replace_with_original,
            ],
            size=len(sentence_tokens),
        ).nonzero()[1]
        token_replacement = (
            (how_to_mask == 0) * self.mask_id
            + (how_to_mask == 1) * self.get_valid_random_tokens(len(sentence_tokens))
            + (how_to_mask == 2) * sentence_tokens
        )
        return np.where(mask_mask, token_replacement, sentence_tokens)

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
    def __init__(self, rng=random, use_dummy_dataset=False):
        self.examples_buffer = []
        self.dataset_wiki = load_dataset(
            "wikipedia", f"20220301.{'simple' if use_dummy_dataset else 'en'}"
        )["train"]
        self.dataset_book = (
            load_dataset("bookcorpus")["train"]
            if not use_dummy_dataset
            else self.dataset_wiki
        )
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
        assert isinstance(processor, BERTSentenceProcessor) or isinstance(
            processor, GPTSentenceProcessor
        )
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

    def __init__(
        self,
        pdataset: ProcessedDataset,
        device: torch.device,
        batch_size: int,
        seq_length: int,
        num_workers: int = 8,
        seed: int = 42,
        model_type: str = "bert",
        dataset_type: Literal["wikibook", "c4"] = "wikibook",
        dataset_split: str = "train",
    ):
        self.device = device
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.sequence_length = seq_length

        if self.model_type == "bert":
            collate_fn = ProcessedBERTBatch
        elif self.model_type == "gpt":
            collate_fn = ProcessedGPTBatch
        else:
            raise ValueError(
                f"Unknown model type in ProcessedDatasetWrapper: {self.model_type}"
            )

        if dataset_type == "wikibook":
            pdataset = ParallelCompatibleDataset(
                pdataset, batch_size=batch_size, seed=seed
            )
        elif dataset_type == "c4":
            pdataset = C4Dataset(seq_length, batch_size, dataset_split)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        self.dataloader = iter(
            DataLoader(
                pdataset,
                num_workers=num_workers,
                batch_size=batch_size,
                collate_fn=collate_fn,
                shuffle=False,  # WikiBookDataset already shuffles
            )
        )

    def get_batch(self) -> ProcessedBatch:
        return next(self.dataloader).to(self.device)


def get_processed_dataset(
    batch_size: int,
    max_total_length: int,
    mask_percent: float,
    device: torch.device,
    num_workers: int,
    seed: int,
    model_type: Literal["bert", "gpt"] = "bert",
    dataset_type: Literal["wikibook", "c4"] = "wikibook",
    use_dummy_dataset: bool = False,
    dataset_split: str = "train",
) -> wikibookdata.ProcessedDatasetWrapper:
    if dataset_type == "wikibook":
        raw_dataset = wikibookdata.WikiBookDataset(use_dummy_dataset=use_dummy_dataset)
        if model_type == "bert":
            processor = wikibookdata.BERTSentenceProcessor(
                max_total_length=max_total_length,
                mask_percent=mask_percent,
            )
        elif model_type == "gpt":
            processor = wikibookdata.GPTSentenceProcessor(
                max_total_length=max_total_length,
            )

        dataset = wikibookdata.ProcessedDataset(raw_dataset, processor)
    else:
        dataset = None

    dataset_wrapper = wikibookdata.ProcessedDatasetWrapper(
        pdataset=dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        model_type=model_type,
        dataset_type=dataset_type,
        dataset_split=dataset_split,
        seq_length=max_total_length,
    )

    # In case of GPT, log an example sequence for a possible inspection
    if model_type == "gpt":
        print("Logging example batch...")
        run = get_current_logger().instance_logger
        batch = dataset_wrapper.get_batch()

        t = GPT2Tokenizer.from_pretrained("gpt2", additional_special_tokens=["<sequence_sep>"])
        num_to_log = 5
        for i in range(num_to_log):
            run[f"example_sequence/seq{i}/text"] = t.decode(batch.tokens[i])
            run[f"example_sequence/seq{i}/target_text"] = t.decode(
                batch.target_tokens[i]
            )
        del batch, t
        print("Logged example batch.")

    return dataset_wrapper
