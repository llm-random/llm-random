import itertools
from typing import List
import unittest

from model import C4Dataset, _process_document
import unittest
from unittest.mock import Mock, patch
import random


class TestLoadingData(unittest.TestCase):

    def test__process_document(self):
        EOT = "<|endoftext|>"

        token_map = {
            "the": 1,
            "dog": 2,
            "cat": 3,
            "ran": 4,
            "sat": 5,
            EOT: 100,
        }

        def mock_encode(text: str, truncation=False, max_length=None) -> List[int]:
            words = text.split()
            tokens = [token_map.get(word, 0) for word in words]  # 0 for unknown words
            return tokens

        test_cases = [
            {"input": {"text": "the dog ran "}, "expected_tokens": [1, 2, 4, 100]},
            {"input": {"text": ""}, "expected_tokens": [100]},
            {
                "input": {"text": f"cat {EOT} sat "},
                "expected_tokens": [3, 100, 5, 100],
            },
        ]

        for case in test_cases:
            document = {"text": case["input"]["text"]}
            result = _process_document(document, mock_encode, EOT)
            self.assertEqual(
                result["tokens"],
                case["expected_tokens"],
                "Failed on case: " + str(case),
            )


class TestC4Dataset(unittest.TestCase):
    def setUp(self):
        self.sequence_length = 5
        self.seed = 42  # dummy seed for reproducibility
        random.seed(self.seed)

        self.mock_tokens = [
            {"tokens": [1, 2, 100]},
            {"tokens": [12, 13, 14, 15, 16, 100]},
            {"tokens": [22, 23, 24, 25, 100]},
            {"tokens": [32, 100]},
            {"tokens": [42, 43, 44, 45, 46, 47, 100]},
            {"tokens": [52, 53, 54, 55, 56, 100]},
            {"tokens": [62, 100]},
            {"tokens": [72, 73, 74, 75, 76, 100]},
        ]

        self.mock_data_generator = Mock()
        self.mock_data_generator.__iter__ = Mock(return_value=iter(self.mock_tokens))

    @patch("random.Random.randint")
    @patch("model.C4Dataset._load_dataset")
    def test_sample_packer(self, mock_load, mock_randint):
        with patch.dict("os.environ", {"WORLD_SIZE": "1", "RANK": "0"}):

            num_samples = 4
            mock_randint.side_effect = [1, 0, 1, 4]  # Define sequence of random numbers

            dataset = C4Dataset(
                sequence_length=self.sequence_length,
                path="dummy_path",
                seed=self.seed,
                tokenizer="dummy_tokenizer",
            )

            dataset.data_generator = self.mock_data_generator

            returned_samples = list(itertools.islice(iter(dataset), num_samples))

            expected_samples = [
                [2, 100, 12, 13, 14],
                [22, 23, 24, 25, 100],
                [100, 42, 43, 44, 45],
                [56, 100, 62, 100, 72],
            ]

            self.assertListEqual(returned_samples, expected_samples)

    @patch("random.Random.randint")
    @patch("model.C4Dataset._load_dataset")
    def test_iter_with_different_ranks(self, mock_load, mock_randint):
        expected_samples = {
            0: [[2, 100, 12, 13, 14], [100, 42, 43, 44, 45]],
            1: [[22, 23, 24, 25, 100], [56, 100, 62, 100, 72]],
        }

        for rank in [0, 1]:
            self.mock_data_generator = Mock()  # Reset the mock data generator
            self.mock_data_generator.__iter__ = Mock(
                return_value=iter(self.mock_tokens)
            )
            with patch.dict("os.environ", {"WORLD_SIZE": "2", "RANK": str(rank)}):
                num_samples = 2
                mock_randint.side_effect = [1, 0, 1, 4]

                dataset = C4Dataset(
                    sequence_length=self.sequence_length,
                    path="dummy_path",
                    seed=self.seed,
                    tokenizer="dummy_tokenizer",
                    world_size_independent=True,
                )

                dataset.data_generator = self.mock_data_generator

                returned_samples = list(itertools.islice(iter(dataset), num_samples))
                self.assertListEqual(returned_samples, expected_samples[rank])
