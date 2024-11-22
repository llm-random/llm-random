from abc import ABC, abstractmethod
import itertools
import random
from typing import Callable, Iterator, List, Optional, Tuple
from attr import define
import regex as re
import spacy

import numpy as np
import torch
from torch.utils.data import IterableDataset

from lizrd.text.datasets import AbstractDataset
from lizrd.text.packers import AbstractPacker, take_circular
from lizrd.text.tokenizers import AbstractTokenizer
from research.mole.utils.data import LLMMetaExample

EOT_POS_TAG = "EOT"
# N_GROUPS = 8
# pos_grouped = {
#  'NOUN': 0,
#  'PUNCT': 1,
#  'NUM': 1,
#  'SPACE': 1,
#  'SYM': 1,
#  'X': 1,
#  'INTJ': 1,
#  EOT_TAG: 1,
#  'VERB': 2,
#  'AUX': 2,
#  'DET': 3,
#  'PRON': 3,
#  'ADP': 4,
#  'PART': 4,
#  'ADJ': 5,
#  'ADV': 5,
#  'PROPN': 6,
#  'CCONJ': 7,
#  'SCONJ': 7,
# }

def encode_with_pos(sentence, tokenizer, spacy_nlp) -> Tuple[list[int], list[str]]:
    spacy_tokens = spacy_nlp(sentence)

    pretokenized = []
    pretokenized_pos = []
    leading_ws = ""
    for t in spacy_tokens:
        pretokenized.append(leading_ws+t.text)
        leading_ws = t.whitespace_
        pretokenized_pos.append(t.pos_)
    
    ids = []
    poss = []
    for pt, pos in zip(pretokenized, pretokenized_pos):
        n_ids = tokenizer.encode(pt)
        ids.extend(n_ids)
        poss.extend([pos for _ in range(len(n_ids))])
    
    assert len(ids) == len(poss)
    return ids, poss

def simple_split(sentence) -> list[str]:
    return re.findall(r'\S+\s*', sentence)

class GPTMetaPOSPacker(
    AbstractPacker,
):
    def __init__(
        self,
        sequence_length: int,
        dataset_maker: AbstractDataset,
        tokenizer_maker: Callable[[], AbstractTokenizer],
        pos_grouped: dict,
        n_experts: int,
        seed: Optional[int] = None,
        
    ):
        super().__init__(
            sequence_length,
            dataset_maker,
            tokenizer_maker,
            seed=seed,
        )
        self.spacy_nlp = spacy.load("en_core_web_sm")
        # self.pos_grouped = {
        #     'NOUN': [0, 9],
        #     'PUNCT': [9, 14],
        #     'VERB': [14, 19],
        #     'ADP': [19, 24],
        #     'PROPN': [24, 28],
        #     'DET': [28, 32],
        #     'ADJ': [32, 35],
        #     'PRON': [35, 38],
        #     'AUX': [38, 40],
        #     'ADV': [40, 42],
        #     'CCONJ': [42, 44],
        #     'PART': [44, 45],
        #     'NUM': [45, 46],
        #     'EOT': [46, 47],
        #     'SPACE': [46, 47],
        #     'SYM': [46, 47],
        #     'X': [46, 47],
        #     'INTJ': [46, 47],
        #     'SCONJ': [47, 48],
        # }
        # self.pos_grouped = {
        #  'NOUN': [0, 1],
        #  'PUNCT': [1, 2],
        #  'NUM': [1, 2],
        #  'SPACE': [1, 2],
        #  'SYM': [1, 2],
        #  'X': [1, 2],
        #  'INTJ': [1, 2],
        #  EOT_POS_TAG: [1, 2],
        #  'VERB': [2, 3],
        #  'AUX': [2, 3],
        #  'DET': [3, 4],
        #  'PRON': [3, 4],
        #  'ADP': [4, 5],
        #  'PART': [4, 5],
        #  'ADJ': [5, 6],
        #  'ADV': [5, 6],
        #  'PROPN': [6, 7],
        #  'CCONJ': [7, 8],
        #  'SCONJ': [7, 8],
        # }
        self.pos_grouped = pos_grouped
        self.n_experts = self.__count_groups_and_validate()
        assert self.n_experts == n_experts

    def __count_groups_and_validate(self):
        c = 0
        seen = []
        # assert list(self.pos_grouped.values())[0][0] == 0 #dev
        for group in self.pos_grouped.values():
            seen.extend(list(range(group[0], group[1])))
        for i in range(len(set(seen))):
            assert i in seen
        return len(set(seen))

    def __get_pos_multitarget(self, pos) -> list:#-> torch.Tensor:
        target = torch.zeros(self.n_experts, dtype=torch.float32)
        ran = self.pos_grouped[pos]
        target[ran[0]:ran[1]] = 1/(ran[1]-ran[0])
        return list(target)


    def get_sample(self) -> LLMMetaExample:
        """
        Sample examples from the dataset until we reach the desired sequence length.
        """
        eot_id = self.tokenizer.eot_id
        assert eot_id is not None

        words_buffer: List[int] = []
        document_lengths: List[int] = []

        while True:
            document = self.dataset.get_document()
            words = simple_split(document)
            words_buffer.extend(words + ["<|endoftext|>"])

            document_lengths.append(len(words) + 1)
            if (sum(document_lengths) - max(document_lengths)) > self.sequence_length:
                break

        sample_start = self.py_rng.randint(0, len(words_buffer) - 1)
        sample_end = sample_start + int(self.sequence_length/1)

        input_words = list(take_circular(words_buffer, sample_start, sample_end))
        input_sentence = "".join(input_words)

        input_docs = input_sentence.split("<|endoftext|>")

        buffer: List[int] = []
        token_metadata_buffer: List[str] = []
        calculate_loss: List[int] = []

        for doc in input_docs:
            tokens, t_pos = encode_with_pos(doc, self.tokenizer.tokenizer, self.spacy_nlp)
            buffer.extend(tokens + [eot_id])
            # t_metadata = [int(self.pos_grouped[t_m]) for t_m in t_metadata]
            token_metadata_buffer.extend(t_pos + [EOT_POS_TAG])
        
        input_ids = buffer[:self.sequence_length]
        t_poss = token_metadata_buffer[:self.sequence_length]
        target_ids = buffer[1:self.sequence_length+1]
        router_target_bias = [self.__get_pos_multitarget(t_pos) for t_pos in t_poss]
        # one_hot_exp_groups = torch.nn.functional.one_hot(torch.tensor(ids_exp_groups), num_classes=N_GROUPS).type(torch.float32)
        
        if len(input_ids) != self.sequence_length:
            return self.get_sample()

        assert len(input_ids) == len(target_ids)
        assert len(input_ids) == len(t_poss)
        assert len(input_ids) == self.sequence_length
        calculate_loss = [1] * len(target_ids)
   
        return LLMMetaExample(input_ids, router_target_bias, target_ids, calculate_loss)
