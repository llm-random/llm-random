from dataclasses import dataclass
from attr import define
import numpy as np


from transformers import BertTokenizer, GPT2Tokenizer


@dataclass
class ProcessedGPTExample(object):
    tokens: list[int]
    non_padded_mask: list[int]
    target_tokens: list[int]


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
        tokens = self.tokenize_text(sentence)
        tokens, non_padded_mask = self.pad_tokens(tokens)
        target_tokens = tokens[1:] + [self.end_token_id]
        return ProcessedGPTExample(tokens, non_padded_mask, target_tokens)

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


class ProcessedBERTExample(object):
    def __init__(self, sentence, processor):
        self.tokens = processor.tokenize_text(sentence)
        self.tokens = processor.pad_tokens(self.tokens)
        special_token_mask = processor.special_token_mask(self.tokens)
        self.mask_mask = processor.get_mask_mask(special_token_mask)
        self.masked_tokens = processor.mask_tokens(self.tokens, self.mask_mask)


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
