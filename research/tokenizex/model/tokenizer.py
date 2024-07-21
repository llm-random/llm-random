import copy
import re
import numpy as np
from typing import List, Tuple

from transformers import GPT2Tokenizer
from lizrd.text.tokenizers import AbstractTokenizer, disable_tokenizer_warnings


class ReversedGPT2Tokenizer:
    def __init__(self, rev_tokenizer:GPT2Tokenizer):
        self.__rev_tokenizer:GPT2Tokenizer = rev_tokenizer

    def prepare_for_tokenization(self, text):
        res_text, aux_info = self.__rev_tokenizer.prepare_for_tokenization(text[::-1])
        return (res_text[::-1], aux_info)

    def encode(self, text):
        if isinstance(text, list):
            rtext = [t[::-1] for t in text]
            e = self.__rev_tokenizer.encode(rtext[::-1])
            return e[::-1]
        e = self.__rev_tokenizer.encode(text[::-1])
        return e[::-1]
    
    def decode(self, tokens):
        d = self.__rev_tokenizer.decode(tokens[::-1])
        return d[::-1]
    
    def tokenize(self, txt):
        return [e[::-1] for e in self.__rev_tokenizer.tokenize(txt[::-1])[::-1]]
    
    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.__rev_tokenizer.convert_tokens_to_ids(tokens[::-1])
        rev = []
        for tok in tokens:
            rtok = tok
            rtok = rtok[::-1]
            rev.append(rtok)
        return self.__rev_tokenizer.convert_tokens_to_ids(rev[::-1])[::-1]

    def get_vocab(self):
        vocab = self.__rev_tokenizer.get_vocab()
        rvocab = {k[::-1]:v for k,v in vocab.items()}
        return rvocab
    
    @property
    def model_max_length(self):
        return self.__rev_tokenizer.model_max_length
    
    @model_max_length.setter
    def model_max_length(self, v):
        self.__rev_tokenizer.model_max_length = v

    @property
    def eos_token(self):
        return self.__rev_tokenizer.eos_token[::-1]


class TokenizexTokenizer(AbstractTokenizer):
    """Best use with pretrained reversed tokenizer - trailing spaces required!"""
    VOCAB_SIZE = 50257
    HEAD_VOCAB_SIZE = (
        256 + 1
    )  # def have to consider special tokens, like eot token can be seen in GPT packer etc.

    def __init__(self):

        self.tokenizer: ReversedGPT2Tokenizer = ReversedGPT2Tokenizer(
            GPT2Tokenizer.from_pretrained("research/tokenizex/model/reversed_tokenizer")
        )
        disable_tokenizer_warnings(self.tokenizer)  # dev

        self.eot_id = self.tokenizer.convert_tokens_to_ids(
            "<|endoftext|>"
        )
        self.eot_id_target = 0

        self._head_cast = self.reorder_atom_tokens(self.tokenizer)
        self._head_cast[self.eot_id_target] = self.eot_id
        assert list(self._head_cast.keys()) == list(self._head_cast.values())   # Ose provided pretrained tokenizer,
        assert self.eot_id_target == self.eot_id                                # or implement casting in target/head decoding.

        assert isinstance(self.eot_id, int)
        assert isinstance(self.eot_id_target, int)

    def prepare_for_tokenization(self, text):
        text = self.tokenizer.prepare_for_tokenization(text)[0]
        text = self.tokenizer.decode(self.tokenizer.encode(text))
        text = re.sub(r"\s+\.", ".", text)
        text = re.sub(r"\s+\'", "'", text)
        text = re.sub(r"\s+\,", ",", text)
        text = re.sub(r"\s+\!", "!", text)
        text = re.sub(r"\s+\?", "?", text)
        text = re.sub(r"\n+\n", "\n", text)
        return text

    def create_ids_pos_mask(self, ids_pos):
        res = []
        mask = np.eye(len(ids_pos))
        for i, (tokids, m) in enumerate(zip(ids_pos, mask)):
            res.append(tokids[-1])
            for ie in tokids[:-1]:
                is_added = True if res[:-1] == [] else False
                for j, je in enumerate(res[:-1]):
                    if ie == je:
                        m[j] = 1
                        is_added = True
                        break
                if not is_added:
                    raise Exception("target len != input len")  # dev !!!!!!!!!!!
        return np.array(res)[:, 0], np.array(res)[:, 1], mask

    def lazy_past_tokens_gather(self, atokens):
        past_ids_pos = []
        for i in range(1, len(atokens) + 1):
            past_ids_pos.append(atokens[:i])
        return past_ids_pos

    def past_tokens_gather(self, atokens):
        past_ids_pos = []
        for i in range(1, len(atokens) + 1):
            pdec = self.tokenizer.decode(atokens[:i])
            if "�" in pdec:
                return self.lazy_past_tokens_gather(atokens)
            penc = self.tokenizer.encode(pdec)
            past_ids_pos.append(penc)
        return past_ids_pos

    # def tokenize_atom(self, txt:str):
    #     atxt = "".join(self.tokenizer.tokenize(txt))
    #     atxt = list(atxt)
    #     eatxt = self.tokenizer.encode(atxt)
    #     return eatxt
    
    def tokenize_atom(self, txt, atomize:np.array = None):
        txt_words = self.split_txt(txt)
        if atomize is None:
            atomize = np.ones(len(txt_words))
        assert len(txt_words) == len(atomize)
        token_atoms = []
        for word, atomize in zip(txt_words, atomize):
            word_tokenized = self.tokenizer.tokenize(word)
            if atomize:
                token_atoms.extend(word_tokenized)
            else:
                token_atoms.extend([atok[0] for atok in word_tokenized])
        atxt = "".join(token_atoms)
        atxt = list(atxt)
        eatxt = self.tokenizer.encode(atxt)
        return eatxt

    def tokenizex_encode_full(self, txt):
        tenc = self.tokenize_atom(txt)
        res = self.past_tokens_gather(tenc)
        tok_pos = [list(zip(e, np.arange(len(e)))) for e in res]
        e, p, m = self.create_ids_pos_mask(tok_pos)
        return e, p, m

    # def split_txt(self, txt): #dev fix for trailing spaces
    #     txtw = [" " + e for e in txt.split(" ")]
    #     if len(txtw[0]) == 1:
    #         del txtw[0]
    #     else:
    #         txtw[0] = txtw[0][1:]
    #     return txtw

    def _split_txt_ap(self, txt):
        txtw = ["'" + e for e in txt.split("'")]
        if len(txtw[0]) == 1:
            del txtw[0]
        else:
            txtw[0] = txtw[0][1:]
        return txtw
    
    def split_txt(self, txt):
        txtw = [e + " " for e in txt.split(" ")]
        if "'" in txt:
            txtw_ap = []
            for e in txtw:
                if "'" in e:
                    txtw_ap.extend(self._split_txt_ap(e))
                else:
                    txtw_ap.append(e)
            txtw = txtw_ap
        if len(txtw[-1]) == 1:
            del txtw[-1]
        else:
            txtw[-1] = txtw[-1][:-1]
        return txtw
    
    def _full_tokens(self, w):
        ids = self.tokenizer.encode(w)
        mask = np.eye(len(ids))
        pos = np.arange(len(ids))
        return ids, pos, mask

    def tokenizex_encode(
        self, txt: str, splitted_words
    ) -> Tuple[List[int], List[int], List[List[int]]]:
        res_ids = []
        res_pos = []
        partial_masks = []
        pos_len = 0
        txtw = self.split_txt(txt)

        if splitted_words is None:
            splitted_words = np.ones(len(txtw))

        assert len(splitted_words) == len(txtw)
        for w, split in zip(txtw, splitted_words):
            if not split:
                ids, pos, mask = self._full_tokens(w)
            else:
                ids, pos, mask = self.tokenizex_encode_full(w)
            res_ids.extend(ids)
            res_pos.extend(pos + pos_len)
            pos_len += pos[-1] + 1
            partial_masks.append(mask)
        full_mask = TokenizexTokenizer.sum_masks(len(res_ids), partial_masks)
        return res_ids, res_pos, full_mask

    def text_to_ids_pos_mask(
        self, text: str, splitted_words=None
    ) -> Tuple[List[int], List[int], List[List[int]]]:
        """Returns tids, tpos and tatt_masks in atoms target aware processing order

        :param str text: text for encoding+
        :return Tuple[List[int], List[int], List[List[int]]]: encoding, positions, attention masks
        """        
        text = self.prepare_for_tokenization(text)
        return self.tokenizex_encode(text, splitted_words)

    def text_to_ids(self, text: str, is_target=False, splitted_words=None) -> List[int]:
        """Used mainly for training target, with option for training encodings"""
        if self.tokenizer.eos_token in text:
            raise NotImplementedError("Not implemented special tokens atomization (aka. target ids gen)")
        text = self.prepare_for_tokenization(text)
        if is_target:
            return self.tokenize_atom(text, splitted_words)
        else:
            raise Exception("Only for target tokens")  # dev
            return self.tokenizex_encode(text)[0]
        
    def ids_to_text(self, ids: list[int]) -> List[int]:
        return self.tokenizer.decode(ids)

    def validate_tokenization(self, txt, ids, pos, mask):
        txt = self.prepare_for_tokenization(txt)
        lids = self.tokenize_atom(txt)
        assert len(lids) == len(ids)

    def validate_tokenization_hard(self, txt:str, ids:np.array, pos:np.array, mask:np.array):
        txt = self.prepare_for_tokenization(txt)

        dec = self.tokenizer.decode(ids[mask[-1].astype(bool)])
        assert dec == txt # dev transformer has no order, token-po out of order

        lids = self.tokenize_atom(txt)
        assert len(lids) == len(ids)

        comp = pos[mask[-1].astype(bool)]
        for a, b in zip(comp.astype(int), list(range(len(comp)))):
            assert a == b # dev transformer has no order, token-po out of order

    @staticmethod
    def sum_masks(o_mask_size, masks):
        iy = 0
        ix = 0
        o_mask = np.zeros((o_mask_size, o_mask_size))

        for m in masks:
            ms = len(m)
            o_mask[iy : iy + ms, ix : ix + ms] += m
            o_mask[iy + ms :, ix : ix + ms] = m[-1]
            iy += ms
            ix += ms
        return o_mask
    
    @staticmethod
    def reorder_atom_tokens(tokenizer) -> dict:
        res = {}
        new_order_c = 0
        vocab = tokenizer.get_vocab()
        vocabi = {v:k for k,v in vocab.items()}
        atom_vocab = {tid: tok for tid, tok in vocabi.items() if len(tok) == 1}
        for tid, tok in atom_vocab.items():
            res[new_order_c+1] = tid
            assert tid == new_order_c+1
            new_order_c += 1
        return res

    @staticmethod
    def sum_positions(poses):
        res = []
        res.extend(poses[0])
        sum_slit = 0
        for i in range(1, len(poses)):
            sum_slit += poses[i - 1][-1] + 1
            res.extend(poses[i] + sum_slit)
        return np.array(res)
