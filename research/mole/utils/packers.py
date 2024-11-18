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

EOT_TAG = "EOT"
N_GROUPS = 8
pos_grouped = {
 'NOUN': 0,
 'PUNCT': 1,
 'NUM': 1,
 'SPACE': 1,
 'SYM': 1,
 'X': 1,
 'INTJ': 1,
 EOT_TAG: 1,
 'VERB': 2,
 'AUX': 2,
 'DET': 3,
 'PRON': 3,
 'ADP': 4,
 'PART': 4,
 'ADJ': 5,
 'ADV': 5,
 'PROPN': 6,
 'CCONJ': 7,
 'SCONJ': 7,
}

def encode_with_meta(sentence, tokenizer, spacy_nlp) -> Tuple[list[int], list[str]]:
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

class GPTMetaPacker(
    AbstractPacker,
):
    def __init__(
        self,
        sequence_length: int,
        dataset_maker: AbstractDataset,
        tokenizer_maker: Callable[[], AbstractTokenizer],
        seed: Optional[int] = None,
    ):
        super().__init__(
            sequence_length,
            dataset_maker,
            tokenizer_maker,
            seed=seed,
        )
        self.spacy_nlp = spacy.load("en_core_web_sm")


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

        # print(len(input_sentence)) #dev
        # print(input_sentence) #dev

        input_docs = input_sentence.split("<|endoftext|>")

        buffer: List[int] = []
        token_metadata_buffer: List[str] = []
        calculate_loss: List[int] = []

        for doc in input_docs:
            tokens, t_metadata = encode_with_meta(doc, self.tokenizer.tokenizer, self.spacy_nlp)
            buffer.extend(tokens + [eot_id])
            t_metadata = [int(pos_grouped[t_m]) for t_m in t_metadata]
            token_metadata_buffer.extend(t_metadata + [int(pos_grouped[EOT_TAG])])
        
        input_ids = buffer[:self.sequence_length]
        ids_exp_groups = token_metadata_buffer[:self.sequence_length]
        target_ids = buffer[1:self.sequence_length+1]
        one_hot_exp_groups = torch.nn.functional.one_hot(torch.tensor(ids_exp_groups), num_classes=N_GROUPS).type(torch.float32)
        
        if len(input_ids) != self.sequence_length:
            return self.get_sample()

        assert len(input_ids) == len(target_ids)
        assert len(input_ids) == len(ids_exp_groups)
        assert len(input_ids) == self.sequence_length
        calculate_loss = [1] * len(target_ids)
   
        return LLMMetaExample(input_ids, one_hot_exp_groups, target_ids, calculate_loss)
