from typing import Callable, Optional, List
import numpy as np

from lizrd.text.datasets import AbstractDataset
from lizrd.text.packers import AbstractPacker

from research.tokenizex.model.tokenizer import TokenizexTokenizer
from research.tokenizex.utils.data import TokenizexExample


class TokenizexGPTPacker(AbstractPacker):
    def __init__(
        self,
        sequence_length: int,
        dataset_maker: AbstractDataset,
        tokenizer_maker: Callable[[], TokenizexTokenizer],
        seed: Optional[int] = None,
    ):
        super().__init__(
            sequence_length,
            dataset_maker,
            tokenizer_maker,
            seed=seed,
        )
        self.tokenizer: TokenizexTokenizer

    def get_sample(self) -> TokenizexExample:
        """
        Sample examples from the dataset until we reach the desired sequence length.
        """
        eot_id = self.tokenizer.eot_id
        eot_id_target = self.tokenizer.eot_id_target
        assert eot_id is not None
        assert eot_id_target is not None

        documents_buffer: List[int] = []
        document_lengths: List[int] = []

        while True:
            document = self.dataset.get_document()
            documents_buffer.append(document)
            document_lengths.append(len(document) + 1)
            if (sum(document_lengths) - max(document_lengths)) > self.sequence_length: #dev ?
                break

        sample_start = self.py_rng.randint(0, len(documents_buffer[0]) - 1)
        documents_buffer[0] = documents_buffer[0][sample_start:]

        ids_buffer: List[int] = []
        positions_buffer: List[int] = []
        masks_buffer: List[int] = []
        terget_ids_buffer: List[int] = []
        ids_len = 0
        for document in documents_buffer:
            ids, pos, mask = self.tokenizer.text_to_ids_pos_mask(document)
            tids = self.tokenizer.text_to_ids(document, True)
            
            # appending eot token
            ids.append(eot_id)
            tids.append(eot_id_target)
            for m in mask:
                m.append(0)
            mask.append(mask[-1])
            mask[-1,-1] = 1

            assert len(ids) == len(tids)
            ids_len+=len(ids)
            surpass = ids_len - self.sequence_length - 1
            if surpass > 0:
                ids_buffer.append(ids[:-surpass])
                positions_buffer.append(pos[:-surpass])
                masks_buffer.append(mask[:-surpass, :-surpass])
                terget_ids_buffer.append(tids[:-surpass])
                break
            ids_buffer.append(ids)
            positions_buffer.append(pos)
            masks_buffer.append(mask)
            terget_ids_buffer.append(tids)

        

        ids = np.concatenate(ids_buffer)[:-1]
        pos = np.concatenate(positions_buffer)[:-1]
        tids = np.concatenate(terget_ids_buffer)[1:]
        mask = TokenizexTokenizer.sum_masks(len(ids), masks_buffer)[:-1,:-1]
        calculate_loss = [1] * len(tids)

        assert len(ids) == len(tids)
        assert len(ids) == len(pos)
        assert len(ids) == len(mask[0])

        return TokenizexExample(ids.tolist(), tids.tolist(), calculate_loss, pos, mask)