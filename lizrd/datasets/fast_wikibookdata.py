import random
import re
import torch
from datasets import load_dataset
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from typing import List
import random
import re
import torch
from datasets import load_dataset
from transformers import BertTokenizer
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from typing import List

import os
import pickle


class SentencePair(object):
    def __init__(self, sen1, sen2):
        self.sen1 = sen1
        self.sen2 = sen2
        self.swapped = False

    def swap(self, sentence_pair2):
        self.sen2, sentence_pair2.sen2 = sentence_pair2.sen2, self.sen2
        self.swapped = True
        sentence_pair2.swapped = True


class ProcessedExampleLean:
    def __init__(self, sentence, processor):
        tokens = processor.tokenize_text(sentence)
        self.tokens = processor.pad_tokens(tokens)
        special_token_mask = processor.special_token_mask(self.tokens)
        self.mask_mask = processor.get_mask_mask(special_token_mask)
        self.masked_tokens = processor.mask_tokens(self.tokens, self.mask_mask)


class OriginalProcessedExample:
    def __init__(self, sentence_pair, processor):
        self.sen1 = sentence_pair.sen1
        self.sen2 = sentence_pair.sen2

        self.sen1_tokens = processor.tokenize_text(sentence_pair.sen1)
        self.sen2_tokens = processor.tokenize_text(sentence_pair.sen2)

        self.tokens = processor.join_sentence_tokens(self.sen1_tokens, self.sen2_tokens)
        self.tokens = processor.pad_tokens(self.tokens)

        self.special_token_mask = processor.special_token_mask(self.tokens)
        self.mask_mask = processor.get_mask_mask(self.special_token_mask)
        self.masked_tokens = processor.mask_tokens(self.tokens, self.mask_mask)
        self.swapped = sentence_pair.swapped


class FastBatch:
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
        return np.array(list_of_token_lists)

    def to_(self, device):
        self.tokens = torch.tensor(self.tokens).to(device)
        self.masked_tokens = torch.tensor(self.masked_tokens).to(device)
        self.mask_mask = torch.tensor(self.mask_mask).to(device)
        return self


class OriginalBatch:
    def __init__(self, processed_examples):
        self.sen1_tokens = [example.sen1_tokens for example in processed_examples]
        self.sen2_tokens = [example.sen2_tokens for example in processed_examples]

        self.tokens = self._make_tensor(
            [example.tokens for example in processed_examples]
        )
        self.special_token_mask = self._make_tensor(
            [example.special_token_mask for example in processed_examples]
        )
        self.mask_mask = self._make_tensor(
            [example.mask_mask for example in processed_examples]
        )
        self.masked_tokens = self._make_tensor(
            [example.masked_tokens for example in processed_examples]
        )
        self.swapped = self._make_tensor(
            [example.swapped for example in processed_examples]
        )
        assert self.tokens.shape == self.masked_tokens.shape
        assert self.tokens.shape == self.special_token_mask.shape
        assert self.tokens.shape == self.mask_mask.shape
        assert self.swapped.shape == (len(processed_examples),)

    def _make_tensor(self, list_of_token_lists):
        return np.array(list_of_token_lists)

    def to_(self, device):
        self.tokens = torch.tensor(self.tokens).to(device)
        self.special_token_mask = torch.tensor(self.special_token_mask).to(device)
        self.mask_mask = torch.tensor(self.mask_mask).to(device)
        self.masked_tokens = torch.tensor(self.masked_tokens).to(device)
        self.swapped = torch.tensor(self.swapped).to(device)
        return self


class OriginalDataset(IterableDataset):
    def __init__(self, seed, processor):
        super().__init__()
        self.examples_buffer = []
        self.dataset_wiki = load_dataset("wikipedia", "20220301.en")["train"]
        self.dataset_book = load_dataset("bookcorpus")["train"]
        self.seed = seed
        self.processor = processor

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

    def _refill_buffer_lean(self):
        while len(self.examples_buffer) <= self.buffer_refill_to:
            self._add_examples_lean(self._get_random_document())
        self.rng.shuffle(self.examples_buffer)

    def _add_examples_lean(self, param):
        """This version simply filters out all sentences that are too short, then adds all remaining sentences to the buffer."""

        document_sentences = [
            sentence for sentence in param if len(sentence) > self.min_sentence_length
        ]
        self.examples_buffer.extend(document_sentences)

    def get_example_lean(self):
        if len(self.examples_buffer) <= self.buffer_refill_from:
            self._refill_buffer_lean()
        example = self.examples_buffer.pop()
        return OriginalProcessedExample(example, self.processor)

    def _get_random_document(self):
        if self.rng.random() < self.wikipedia_chance:
            document_text = self.dataset_wiki[
                self.rng.randint(0, len(self.dataset_wiki) - 1)
            ]["text"]
            dots_to_newlines = re.sub(r"\.\s", "\n", document_text)
            document_sentences = re.sub(r"\n+", "\n", dots_to_newlines).split("\n")
            assert isinstance(document_sentences, list)
            assert isinstance(document_sentences[0], str)
        else:
            linebegin = self.rng.randint(
                0, len(self.dataset_wiki) - 1 - self.bookcorpus_lines
            )
            lineend = linebegin + self.bookcorpus_lines
            document_sentences = self.dataset_book[linebegin:lineend]
            document_sentences = [sentence for sentence in document_sentences]
            assert isinstance(document_sentences, list)
            assert isinstance(document_sentences[0], str)
        return document_sentences

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.rng = random.Random(self.seed)
        else:  # in a worker process
            self.rng = random.Random(self.seed + worker_info.id)
        while True:
            yield self.get_example_lean()


class FastDataset(IterableDataset):
    def __init__(self, seed, processor):
        super().__init__()
        self.examples_buffer = []
        self.dataset_wiki = load_dataset("wikipedia", "20220301.en")["train"]
        self.dataset_book = load_dataset("bookcorpus")["train"]
        self.seed = seed
        self.processor = processor

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

    def _refill_buffer_lean(self):
        while len(self.examples_buffer) <= self.buffer_refill_to:
            self._add_examples_lean(self._get_random_document())
        self.rng.shuffle(self.examples_buffer)

    def _add_examples_lean(self, param):
        """This version simply filters out all sentences that are too short, then adds all remaining sentences to the buffer."""

        document_sentences = [
            sentence for sentence in param if len(sentence) > self.min_sentence_length
        ]
        self.examples_buffer.extend(document_sentences)

    def get_example_lean(self):
        if len(self.examples_buffer) <= self.buffer_refill_from:
            self._refill_buffer_lean()
        example = self.examples_buffer.pop()
        return ProcessedExampleLean(example, self.processor)

    def _get_random_document(self):
        if self.rng.random() < self.wikipedia_chance:
            document_text = self.dataset_wiki[
                self.rng.randint(0, len(self.dataset_wiki) - 1)
            ]["text"]
            dots_to_newlines = re.sub(r"\.\s", "\n", document_text)
            document_sentences = re.sub(r"\n+", "\n", dots_to_newlines).split("\n")
            assert isinstance(document_sentences, list)
            assert isinstance(document_sentences[0], str)
        else:
            linebegin = self.rng.randint(
                0, len(self.dataset_wiki) - 1 - self.bookcorpus_lines
            )
            lineend = linebegin + self.bookcorpus_lines
            document_sentences = self.dataset_book[linebegin:lineend]
            document_sentences = [sentence for sentence in document_sentences]
            assert isinstance(document_sentences, list)
            assert isinstance(document_sentences[0], str)
        return document_sentences

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.rng = random.Random(self.seed)
        else:  # in a worker process
            self.rng = random.Random(self.seed + worker_info.id)
        while True:
            yield self.get_example_lean()


class OriginalDataset(IterableDataset):
    def __init__(self, seed, processor):
        super().__init__()
        self.examples_buffer = []
        self.dataset_wiki = load_dataset("wikipedia", "20220301.en")["train"]
        self.dataset_book = load_dataset("bookcorpus")["train"]
        self.seed = seed
        self.processor = processor

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

    def _refill_buffer(self):
        while len(self.examples_buffer) <= self.buffer_refill_to:
            last_len = len(self.examples_buffer)
            self._add_examples(self._get_random_document())
        random.shuffle(self.examples_buffer)

    def _add_examples(self, document_sentences):
        emptysentencelength = 5
        document_sentences.append(
            "a" * emptysentencelength
        )  # hack to ensure last sentences can be added
        good_sentences = []
        for sentence in document_sentences:
            if len(sentence) > self.min_sentence_length:
                good_sentences.append(sentence)
            elif len(sentence.strip()) < emptysentencelength:
                continue
            else:
                if len(good_sentences) % 2 == 1:
                    if random.random() < 0.5:
                        good_sentences.pop()
                    else:
                        good_sentences.pop(0)
                for i in range(0, len(good_sentences), 2):
                    pair = SentencePair(good_sentences[i], good_sentences[i + 1])
                    self.examples_buffer.append(pair)
                good_sentences = []

    def process_batch(self, sentence_pairs):
        return self.processor.process_batch(sentence_pairs)
        # for i in range(0, int(len(sentence_pairs) * self.swap_percent), 2):
        #     sentence_pairs[i].swap(sentence_pairs[i + 1])
        # random.shuffle(sentence_pairs)
        # return ProcessedBatch(
        #     [self.process(sentence_pair) for sentence_pair in sentence_pairs],
        #     device=self.device,
        # )

    def get_example(self):
        if len(self.examples_buffer) <= self.buffer_refill_from:
            self._refill_buffer()
        example = self.examples_buffer.pop()
        return OriginalProcessedExample(example, self.processor)

    def _get_random_document(self):
        if self.rng.random() < self.wikipedia_chance:
            document_text = self.dataset_wiki[
                self.rng.randint(0, len(self.dataset_wiki) - 1)
            ]["text"]
            dots_to_newlines = re.sub(r"\.\s", "\n", document_text)
            document_sentences = re.sub(r"\n+", "\n", dots_to_newlines).split("\n")
            assert isinstance(document_sentences, list)
            assert isinstance(document_sentences[0], str)
        else:
            linebegin = self.rng.randint(
                0, len(self.dataset_wiki) - 1 - self.bookcorpus_lines
            )
            lineend = linebegin + self.bookcorpus_lines
            document_sentences = self.dataset_book[linebegin:lineend]
            document_sentences = [sentence for sentence in document_sentences]
            assert isinstance(document_sentences, list)
            assert isinstance(document_sentences[0], str)
        return document_sentences

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.rng = random.Random(self.seed)
        else:  # in a worker process
            self.rng = random.Random(self.seed + worker_info.id)
        while True:
            yield self.get_batch()


class FastDataloader:
    def collate_fn(self, processed_examples: List[ProcessedExampleLean]):
        if isinstance(self.processor, OriginalSentencePairProcessor):
            return processed_examples[0].to_(self.device)
            # for i in range(0, int(len(sentence_pairs) * self.swap_percent), 2):
            #     sentence_pairs[i].swap(sentence_pairs[i + 1])
            # self.rng.shuffle(sentence_pairs)
            # return OriginalBatch(processed_examples)
        elif isinstance(self.processor, LeanSentencePairProcessor):
            return FastBatch(processed_examples)
        else:
            assert False, "Unknown processor type"

    def __init__(self, batch_size, num_workers, device, seed, processor):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.seed = seed
        self.processor = processor
        dataset = None
        if isinstance(self.processor, LeanSentencePairProcessor):
            dataset = FastDataset(seed, processor)
        else:
            dataset = OriginalDataset(seed, processor)
        self.dataloader = iter(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=self.collate_fn,
            )
        )

    def get_batch(self, device=None):
        return next(self.dataloader).to_(device or self.device)


class OriginalSentencePairProcessor:
    def __init__(self, max_total_length=128, mask_percent=0.15, swap_percent=0.5):
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
        self.swap_percent = swap_percent  # only for batches!

    def process(self, sentence_pair):
        return OriginalProcessedExample(sentence_pair, self)

    def process_batch(self, sentence_pairs):
        for i in range(0, int(len(sentence_pairs) * self.swap_percent), 2):
            sentence_pairs[i].swap(sentence_pairs[i + 1])
        random.shuffle(sentence_pairs)
        return OriginalBatch(
            [self.process(sentence_pair) for sentence_pair in sentence_pairs]
        )

    def tokenize_text(self, sentence_text):
        # note: tokenizer.encode _claims_ to be equivalent. This isn't true.
        return self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(sentence_text)
        )

    def special_token_mask(self, sentence_tokens):
        return np.isin(sentence_tokens, self.special_token_ids)

    def get_mask_mask(self, special_token_mask):
        mask_mask = np.random.binomial(1, self.mask_percent, len(special_token_mask))
        mask_mask = mask_mask.astype(bool)
        mask_mask = np.where(special_token_mask, 0, mask_mask)
        return mask_mask

    def mask_tokens(self, sentence_tokens, mask_mask):
        sentence_tokens = np.where(mask_mask, self.mask_id, sentence_tokens)
        return sentence_tokens

    def join_sentence_tokens(self, sentence_tokens1, sentence_tokens2):
        if len(sentence_tokens1) > self.max_sentence_length:
            sentence_tokens1 = sentence_tokens1[-self.max_sentence_length :]
        if len(sentence_tokens2) > self.max_sentence_length:
            sentence_tokens2 = sentence_tokens2[: self.max_sentence_length]
        return (
            [self.cls_id]
            + sentence_tokens1
            + [self.sep_id]
            + sentence_tokens2
            + [self.sep_id]
        )

    def pad_tokens(self, sentence_tokens):
        if len(sentence_tokens) > self.max_total_length:
            sentence_tokens = sentence_tokens[: self.max_total_length]
        return sentence_tokens + [self.pad_id] * (
            self.max_total_length - len(sentence_tokens)
        )


class LeanSentencePairProcessor:
    def __init__(
        self, max_total_length=128, mask_percent=0.15, swap_percent=0.5, device="cpu"
    ):
        self.device = device
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
        self.swap_percent = swap_percent  # only for batches!

    def process(self, sentence):
        return ProcessedExampleLean(sentence, self)

    def process_batch(self, sentence_pairs):
        for i in range(0, int(len(sentence_pairs) * self.swap_percent), 2):
            sentence_pairs[i].swap(sentence_pairs[i + 1])
        random.shuffle(sentence_pairs)
        return ProcessedBatch(
            [self.process(sentence_pair) for sentence_pair in sentence_pairs],
            device=self.device,
        )

    def process_batch_lean(self, sentences):
        return ProcessedBatchLean(
            [self.process(sentence) for sentence in sentences],
            device=self.device,
        )

    def tokenize_text(self, sentence_text):
        # note: tokenizer.encode _claims_ to be equivalent. This isn't true.
        return self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(sentence_text)
        )

    def special_token_mask(self, sentence_tokens):
        return np.isin(sentence_tokens, self.special_token_ids)

    def get_mask_mask(self, special_token_mask):
        mask_mask = np.random.binomial(1, self.mask_percent, len(special_token_mask))
        mask_mask = mask_mask.astype(bool)
        mask_mask = np.where(special_token_mask, 0, mask_mask)
        return mask_mask

    def mask_tokens(self, sentence_tokens, mask_mask):
        sentence_tokens = np.where(mask_mask, self.mask_id, sentence_tokens)
        return sentence_tokens

    def join_sentence_tokens(self, sentence_tokens1, sentence_tokens2):
        if len(sentence_tokens1) > self.max_sentence_length:
            sentence_tokens1 = sentence_tokens1[-self.max_sentence_length :]
        if len(sentence_tokens2) > self.max_sentence_length:
            sentence_tokens2 = sentence_tokens2[: self.max_sentence_length]
        return (
            [self.cls_id]
            + sentence_tokens1
            + [self.sep_id]
            + sentence_tokens2
            + [self.sep_id]
        )

    def pad_tokens(self, sentence_tokens):
        if len(sentence_tokens) > self.max_total_length:
            sentence_tokens = sentence_tokens[: self.max_total_length]
        return sentence_tokens + [self.pad_id] * (
            self.max_total_length - len(sentence_tokens)
        )
