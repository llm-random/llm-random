from lizrd.core.llm import LLM


import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.models.utils import Collator, pad_and_concat
from tqdm.auto import tqdm


import random
from typing import List, Tuple


class HarnessLM(TemplateLM):
    """
    This class is a wrapper around a LLM that allows it to be used with the lm-eval library. Some methods that are required only by particular benchmarks might not be implemented.
    This code is mostly taken from here: https://github.com/EleutherAI/lm-evaluation-harness/blob/7ad7c5b9d0f1c35c048af0ce8b197ebc2021dbd3/lm_eval/models/huggingface.py#L69 , i.e. it's in most part a copy of the HuggingFace wrapper class from lm-eval.
    """

    def __init__(self, model: LLM, batch_size: int, tokenizer, max_length: int, device):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def eot_token_id(self):
        return self.tokenizer.eot_id

    def generate_until(self, x: list[Instance]):
        raise NotImplementedError

    def loglikelihood_rolling(self, x: list[Instance]):
        raise NotImplementedError

    def tok_encode(self, s: str) -> List[int]:
        return [self.eot_token_id()] + self.tokenizer.text_to_ids(s)

    def _select_cont_toks(
        self, logits: torch.Tensor, contlen: int = None, inplen: int = None
    ) -> torch.Tensor:
        assert (
            contlen and inplen
        ), "Must pass input len and cont. len to select scored logits for causal LM"
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen : inplen]

        return logits

    # _loglikelihood_tokens,
    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        n_padding_requests = 0
        if len(requests) % self.batch_size != 0:
            print("Warning: batch size is not a multiple of the number of requests")
            n_padding_requests = self.batch_size - (len(requests) % self.batch_size)
            for _ in range(n_padding_requests):
                requests.append(random.choice(requests))

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            # group_by="contexts",
            group_by=None,
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = self.batch_size
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        sum_of_chunks = 0
        for chunk in chunks:
            sum_of_chunks += len(chunk)
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                (inplen,) = inp.shape

                # build encoder attn masks
                encoder_attns.append(torch.ones_like(inp))

                cont = torch.tensor(
                    (continuation_enc)[-self.max_length :],
                    # TODO: left-shift these?
                    # TODO: our code assumes we never end up truncating conts for either model type
                    dtype=torch.long,
                    device=self.device,
                )
                (contlen,) = cont.shape

                conts.append(cont)

                padding_len_cont = (
                    max(padding_len_cont, contlen)
                    if padding_len_cont is not None
                    else contlen
                )

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            batched_inps = pad_and_concat(
                padding_len_inp, inps, padding_side="right"
            )  # [batch, padding_len_inp]

            # multi_logits = F.log_softmax(
            #     self._model_call(batched_inps, **call_kwargs), dim=-1
            # )  # [batch, padding_length (inp or cont), vocab]
            multi_logits = F.log_softmax(
                self.model(batched_inps, **call_kwargs), dim=-1
            )  # [batch, padding_length (inp or cont), vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(
                        0
                    )  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()

        res = re_ord.get_original(res)
        if n_padding_requests > 0:
            res = res[:-n_padding_requests]
        return res
