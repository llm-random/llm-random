import numpy as np

from functools import cache
from transformers import GPT2TokenizerFast
from typing import List, Optional

from lizrd.text.tokenizers import AbstractTokenizer, disable_tokenizer_warnings


class TokenizexTokenizer(AbstractTokenizer):
    VOCAB_SIZE = 50257
    HEAD_VOCAB_SIZE = 256 #def have to consider special tokens, like eot token can be seen in GPT packer etc. 

    def __init__(self, atomizer: dict = None):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") #dev GPT2TokenizerFast as base tokenizer
        disable_tokenizer_warnings(self.tokenizer) #dev

        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.atomizer = TokenizexTokenizer.create_atomizer(self.tokenizer) if not atomizer else atomizer
        self.head_cast = TokenizexTokenizer.reorder_atomize_tokens(self.tokenizer)

        assert isinstance(self.eot_id, int)

    def embedding_pos_mask(self, txt):
        tokens = self.tokenizer.encode(txt)
        partial_masks = []
        res_emb = []
        res_pos = []
        pos_len = 0

        for t in tokens:
            emb, pos, mask = self.atomizer[t]
            res_emb.extend(emb)
            res_pos.extend(pos+pos_len)
            pos_len += pos[-1] + 1
            partial_masks.append(mask)
                    
        full_mask = TokenizexTokenizer.sum_masks(len(res_emb), partial_masks)
        return np.array(res_emb), np.array(res_pos), full_mask
    
    def text_to_ids_pos_mask(self, text: str) -> List[int]:
        return self.embedding_pos_mask(text)

    def text_to_ids(self, text: str, is_target = False) -> List[int]:
        """Used mainly for training target, with option for training embedding
        """        
        if is_target:
            emb = []
            for l in text:
                emb.extend(self.tokenizer.encode(l))
            return emb
        else:    
            raise Exception("checking where called") #dev
            return self.embedding_pos_mask(text)[0]
        
    def ids_to_txt(self, ids):
        base_ids = []
        for id in ids:
            base_ids.append(self.head_cast[id])
        return self.tokenizer.decode(base_ids)
        
    @staticmethod
    def _text_sub_emb_pos(self, txt):
        emb_pos = []
        for i in range(1, len(txt)+1):
            et = self.tokenizer.encode(txt[:i])
            emb_pos.append(list(zip(et, list(range(len(et))))))
        return emb_pos
    
    @staticmethod
    def _create_subs_emb_mask(emb_pos):
        total_len = len(emb_pos)
        emb = []
        mask = np.zeros([total_len, total_len])
        for embt, m in zip(emb_pos, mask):
            emb.append(embt[-1])
            m[len(emb)-1] = 1
            for ie in embt[:-1]:
                for j, je in enumerate(emb[:-1]):
                    if je == ie:
                        m[j] = 1
                        break
        return np.array(emb), mask

    @staticmethod
    def _create_emb_pos_mask(txt, tokenizer):
        emb_pos = TokenizexTokenizer._text_sub_emb_pos(txt, tokenizer)
        res, mask = TokenizexTokenizer._create_subs_emb_mask(emb_pos)    
        return res[:, 0], res[:, 1], mask
    
    @staticmethod
    def _process_token(t, tokenizer):
        td = tokenizer.decode(t)
        if len(td)==1 or "�" in td:
            emb = np.array([t]) 
            pos = np.zeros((1))
            mask = np.ones((1,1))
        else:
            emb, pos, mask = TokenizexTokenizer._create_emb_pos_mask(td, tokenizer)
        return emb, pos, mask
    
    @staticmethod
    def create_atomizer(tokenizer: GPT2TokenizerFast):
        """
        Based on given tokenizer returns atomizer for its tokens (cached alternative, _process_token for every token),
        {
            token_id:{
                embedding, 
                positiolan_encoding_positions, 
                attention_mask
            }
        }
        contains each token informations: its atom-tokens, positional encoding positions and attantion mask <- all in tokenizex processing format 

        :param _type_ tokenizer: _description_
        :return dict[int, tuple[np.array, np.array, np.array]]: aka. atomizer, 
        """
        tokenizex_atomizer = {}
        vocab = {v:k for k, v in tokenizer.get_vocab().items()}
        all_tok = vocab.keys()

        for t in all_tok:
            e, p, m = TokenizexTokenizer._process_token(t, tokenizer)
            tokenizex_atomizer[t] = (e, p, m)
            
        return tokenizex_atomizer
    
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
