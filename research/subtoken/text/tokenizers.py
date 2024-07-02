from abc import ABC, abstractmethod
from typing import List, Optional
from transformers import BertTokenizerFast, GPT2TokenizerFast, AutoTokenizer


class AbstractTokenizer(ABC):
    VOCAB_SIZE: int
    MAX_N_BYTES: int
    sequence_separator_id: Optional[int]
    mask_id: Optional[int]
    eot_id: Optional[int]
    blanks_ids: Optional[List[int]]

    @abstractmethod
    def text_to_ids(self, text: str) -> List[int]:
        raise NotImplementedError()


def disable_tokenizer_warnings(hf_tokenizer):
    # set model max length to high number to disable warnings
    # we handle sequence length ourselves
    hf_tokenizer.model_max_length = 100_000


class BertTokenizer(AbstractTokenizer):
    VOCAB_SIZE = 30522

    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        disable_tokenizer_warnings(self.tokenizer)
        self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        assert isinstance(self.mask_id, int)
        self.sequence_separator_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        assert isinstance(self.sequence_separator_id, int)

    def text_to_ids(self, text: str) -> List[int]:
        # TODO: encode or tokenize + convert_tokens_to_ids?
        return self.tokenizer.encode(text)


class GPTTokenizer(AbstractTokenizer):
    VOCAB_SIZE = 50257
    # UNWANTED_TOKENS = {
    #     int(x)
    #     for x in """3880, 8864, 10052, 10097, 10221, 14827, 14950, 15171, 16529, 17174, 19351, 20368, 20727, 22369, 23090, 23193, 23926, 27006, 27193, 27473, 27754, 28542, 28719, 29113, 29146, 29760, 29789, 30210, 30213, 30542, 30899, 30906, 30982, 31576, 32799, 32941, 34400, 35496, 36174, 36573, 36658, 37389, 38093, 39172, 39177, 39753, 39755, 39756, 39757, 40242, 40586, 40800, 40887, 41380, 41436, 41906, 42045, 43453, 43649, 43801, 44436, 44713, 45545, 45706, 46111, 46674, 47232, 47757, 48667, 49129, 49527, 49704""".split(
    #         ", "
    #     )
    # }
    MAX_BYTES_PER_TOKEN = 16

    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        disable_tokenizer_warnings(self.tokenizer)
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        all_tokens_raw_text = self.tokenizer.convert_ids_to_tokens(
            list(range(self.VOCAB_SIZE))
        )
        all_tokens_actual_text = [
            self.tokenizer.convert_tokens_to_string([all_tokens_raw_text[id_]])
            for id_ in range(self.VOCAB_SIZE)
        ]
        self.token_id_to_text = all_tokens_actual_text
        assert isinstance(self.eot_id, int)

    def text_to_ids(self, text: str) -> List[int]:
        # TODO: encode or tokenize + convert_tokens_to_ids?
        return self.tokenizer.encode(text)


class T5Tokenizer(AbstractTokenizer):
    VOCAB_SIZE = 32100
    MAX_N_BYTES = 20

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        disable_tokenizer_warnings(self.tokenizer)
        self.eot_id = 1

        assert isinstance(self.eot_id, int)

    def text_to_ids(self, text: str) -> List[int]:
        # TODO: encode or tokenize + convert_tokens_to_ids?
        return self.tokenizer.encode(text)


class OpenLlama2Tokenizer(AbstractTokenizer):
    VOCAB_SIZE = 32000

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "openlm-research/open_llama_3b_v2"
        )
        disable_tokenizer_warnings(self.tokenizer)
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<s>")[0]

        assert isinstance(self.eot_id, int)

    def text_to_ids(self, text: str) -> List[int]:
        # TODO: encode or tokenize + convert_tokens_to_ids?
        return self.tokenizer.encode(text, add_special_tokens=False)
