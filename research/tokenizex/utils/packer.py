from typing import Callable, Optional, List
import numpy as np

from lizrd.text.datasets import AbstractDataset
from lizrd.text.packers import AbstractPacker

from research.tokenizex.model.tokenizer import TokenizexTokenizer
from research.tokenizex.utils.data import TokenizexExample


class AtomizationManager:
    def __init__(self, packer: "TokenizexGPTPacker", atom_p: float):
        self.atom_p = atom_p
        self.packer = packer
        self.switched_atom_p = None

    def __enter__(self):
        self.switched_atom_p = self.packer.atomization_p
        self.packer.atomization_p = self.atom_p

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.packer.atomization_p = self.switched_atom_p
        self.switched_atom_p = None


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
        self.atomization_p: float = 0.0
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
            document_lengths.append(
                len(self.tokenizer.tokenizer.tokenize(document))
            )
            if (
                sum(document_lengths)
                + len(document_lengths)
                - max(document_lengths)
                - 1
            ) > self.sequence_length:  #dev check
                break

        start = self.py_rng.randint(0, sum(document_lengths) - 1)
        documents_tokenization_buffer = []
        documents_tokenization_lengths = []
        sum_len = 0
        atimization_mask = None
        for i, l in enumerate(document_lengths):
            if (
                sum_len + l > start
            ):  # dev check
                documents_tokenization_buffer.append(
                    documents_buffer[i][sum_len - start :]
                )
                documents_tokenization_lengths.append(
                    len(
                        self.tokenizer.tokenizer.tokenize(
                            documents_buffer[i][sum_len - start :]
                        )
                    )
                )
                for j in range(i + 1, len(documents_buffer) * 2):
                    doc_ix = j % len(documents_buffer)
                    documents_tokenization_buffer.append(documents_buffer[doc_ix])
                    documents_tokenization_lengths.append(document_lengths[doc_ix])
                break
            sum_len += l
        documents_buffer = documents_tokenization_buffer
        document_lengths = documents_tokenization_lengths

        ids_buffer: List[int] = []
        positions_buffer: List[int] = []
        masks_buffer: List[int] = []
        terget_ids_buffer: List[int] = []
        ids_len = 0
        for document, document_len in zip(documents_buffer, document_lengths):
            if document_len > int(self.sequence_length * 1.1):
                document = document[:len("".join(self.tokenizer.tokenizer.tokenize(document)[: int(self.sequence_length * 1.1)]))] #dev FIX 22.07

            document_prep = self.tokenizer.prepare_for_tokenization(document)
            document_words = self.tokenizer.split_txt(document_prep)
            random_array = np.random.rand(len(document_words)) #dev atomize
            atimization_mask = (random_array < self.atomization_p).astype(int) #dev atomize

            ids, pos, mask = self.tokenizer.text_to_ids_pos_mask(document, atimization_mask)
            mask = mask.tolist()  # dev
            tids = self.tokenizer.text_to_ids(document, True, atimization_mask)

            # appending eot token
            ids.append(eot_id)
            pos.append(pos[-1] + 1)  # dev
            tids.append(eot_id_target)
            for m in mask:
                m.append(0)
            mask.append(mask[-1])
            mask[-1][-1] = 1
            mask = np.array(mask)  # dev

            assert len(ids) == len(tids)
            ids_len += len(ids)
            surpass = ids_len - self.sequence_length - 1
            if surpass > 0:
                if surpass == len(ids):
                    break
                ids_buffer.append(ids[:-surpass])
                positions_buffer.append(pos[:-surpass])
                masks_buffer.append(mask[:-surpass, :-surpass])
                terget_ids_buffer.append(tids[:-surpass])
                break
            ids_buffer.append(ids)
            positions_buffer.append(pos)
            masks_buffer.append(mask)
            terget_ids_buffer.append(tids)

        positions_buffer = [np.array(e) for e in positions_buffer]
        for i in range(1, len(positions_buffer)):
            positions_buffer[i] = (
                positions_buffer[i] + positions_buffer[i - 1][-1]
            )

        ids = np.concatenate(ids_buffer)[:-1]
        pos = np.concatenate(positions_buffer)[:-1]
        tids = np.concatenate(terget_ids_buffer)[1:]
        mask = TokenizexTokenizer.sum_masks(len(ids) + 1, masks_buffer)[:-1, :-1]
        calculate_loss = [1] * len(tids)

        tids_def_tokenized = self.tokenizer.tokenizer.tokenize(
            self.tokenizer.tokenizer.decode(tids)
        )
        deftok_byte_scale = self.sequence_length / len(tids_def_tokenized)

        assert len(ids) == self.sequence_length
        assert len(ids) == len(tids)
        assert len(ids) == len(pos)
        assert len(ids) == len(mask[0])

        # mask = np.tril(np.ones(mask.shape)) #dev

        return TokenizexExample(
            ids.tolist(),
            tids.tolist(),
            calculate_loss,
            pos,
            mask.astype(bool),
            deftok_byte_scale,
        )
