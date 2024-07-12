import re
import numpy as np

from functools import cache
from transformers import GPT2Tokenizer
from typing import List, Optional, Tuple

from lizrd.text.tokenizers import AbstractTokenizer, disable_tokenizer_warnings


class TokenizexTokenizer(AbstractTokenizer):
    VOCAB_SIZE = 50257
    HEAD_VOCAB_SIZE = 256 + 1 #def have to consider special tokens, like eot token can be seen in GPT packer etc. 

    def __init__(self):
        self.tokenizer:GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2") #dev GPT2TokenizerFast as base tokenizer
        disable_tokenizer_warnings(self.tokenizer) #dev

        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>") #dev beware!
        self.eot_id_target = 256 #dev beware!

        self.head_cast = TokenizexTokenizer.reorder_atomize_tokens(self.tokenizer) # Head ids cast to default tokenizers atoms ids
        self.head_cast[self.eot_id_target] = self.eot_id

        assert isinstance(self.eot_id, int)
        assert isinstance(self.eot_id_target, int)

    def prepare_for_tokenization(self, text):
        text = self.tokenizer.prepare_for_tokenization(text)[0]
        text = self.tokenizer.decode(self.tokenizer.encode(text))
        text = re.sub(r'\s+\.', '.', text)
        text = re.sub(r"\s+\'", "'", text)
        text = re.sub(r'\s+\,', ',', text)
        text = re.sub(r'\s+\!', '!', text)
        text = re.sub(r'\s+\?', '?', text)
        text = re.sub(r'\n+\n', '\n', text)
        return text
    
    def create_ids_pos_mask(self, ids_pos):
        res = []
        mask = np.eye(len(ids_pos))
        for tokids, m in zip(ids_pos, mask):
            res.append(tokids[-1])
            for ie in tokids[:-1]:
                is_added = True if res[:-1] == [] else False
                for j, je in enumerate(res[:-1]):
                    if ie == je:
                        m[j] = 1
                        is_added = True
                        break
                if not is_added:
                    raise Exception("target len != input len") #dev
        return np.array(res)[:,0], np.array(res)[:,1], mask
    
    def lazy_past_tokens_gather(self, atokens):
        past_ids_pos = []
        for i in range(1, len(atokens)+1):
            past_ids_pos.append(atokens[:i])
        return past_ids_pos

    def past_tokens_gather(self, atokens):
        past_ids_pos = []
        for i in range(1, len(atokens)+1):
            pdec = self.tokenizer.decode(atokens[:i])
            if '�' in pdec:
                return self.lazy_past_tokens_gather(atokens)
            penc = self.tokenizer.encode(pdec)
            past_ids_pos.append(penc)
        return past_ids_pos
    
    def tokenize_atom(self, txt):
        atxt = "".join(self.tokenizer.tokenize(txt))
        atxt = list(atxt)
        eatxt = self.tokenizer.encode(atxt)
        return eatxt

    def tokenizex_encode_full(self, txt):
        tenc = self.tokenize_atom(txt)
        res = self.past_tokens_gather(tenc)
        tok_pos = [list(zip(e, np.arange(len(e)))) for e in res]
        e, p, m = self.create_ids_pos_mask(tok_pos)
        return e, p, m
    
    def split_txt(self, txt):
        txtw = [" " + e for e in txt.split(" ")]
        if len(txtw[0]) == 1:
            del txtw[0]
        else:
            txtw[0] = txtw[0][1:]
        return txtw

    def tokenizex_encode(self, txt: str) -> Tuple[List[int], List[int], List[List[int]]]:
        res_ids = []
        res_pos = []
        partial_masks = []
        pos_len = 0
        txtw = self.split_txt(txt)
        
        for w in txtw:
            ids, pos, mask = self.tokenizex_encode_full(w)
            res_ids.extend(ids)
            res_pos.extend(pos+pos_len)
            pos_len += pos[-1] + 1
            partial_masks.append(mask)   
        full_mask = TokenizexTokenizer.sum_masks(len(res_ids), partial_masks)
        return res_ids, res_pos, full_mask
    
    def text_to_ids_pos_mask(self, text: str) -> Tuple[List[int], List[int], List[List[int]]]:
        text = self.prepare_for_tokenization(text)
        return self.tokenizex_encode(text)

    def text_to_ids(self, text: str, is_target = False) -> List[int]:
        """Used mainly for training target, with option for training encodings
        """        
        text = self.prepare_for_tokenization(text)
        if is_target:
            return self.tokenize_atom(text)
        else:    
            raise Exception("checking where called") #dev
            return self.tokenizex_encode(text)[0]
        
    def validate_tokenization(self, txt, ids, pos, mask):
        txt = self.prepare_for_tokenization(txt)

        # dec = self.tokenizer.decode(ids[mask[-1].astype(bool)])
        # assert dec == txt # dev transformer has no order, token-po out of order

        lids = self.tokenize_atom(txt)
        assert len(lids) == len(ids)

        # comp = pos[mask[-1].astype(bool)]
        # for a, b in zip(comp.astype(int), list(range(len(comp)))):
        #     assert a == b # dev transformer has no order, token-po out of order

    @staticmethod
    def sum_masks(o_mask_size, masks):
        iy = 0
        ix = 0
        o_mask = np.zeros((o_mask_size, o_mask_size))

        for m in masks:
            ms = len(m)
            o_mask[iy:iy+ms, ix:ix+ms] += m
            o_mask[iy+ms:, ix:ix+ms] = m[-1]
            iy+=ms
            ix+=ms
        return o_mask
    
    @staticmethod
    def reorder_atomize_tokens(tokenizer) -> dict:
        res = {}
        new_order_c = 0
        vocab = tokenizer.get_vocab()
        for e in vocab:
            if len(e) == 1:
                res[new_order_c] = vocab[e]
                new_order_c += 1
        return res
    
    @staticmethod
    def sum_positions(poses):
        res = []
        res.extend(poses[0])
        sum_slit = 0
        for i in range(1, len(poses)):
            sum_slit += poses[i-1][-1] + 1
            res.extend(poses[i] + sum_slit)
        return np.array(res)