from lizrd.support.test_utils import GeneralTestCase
from research.blanks.tokenizers import MAX_BLANKS, BlankTokenizer


class TestUtils(GeneralTestCase):
    def test_blank_tokens_ids(self):
        """Tests that the blank tokens ids are correct. This test is potentially heavy - if tests become unbearably slow, consider marking it as with @heavy_test."""
        tokenizer = BlankTokenizer()
        static_ids = BlankTokenizer.BLANK_IDS
        actual_blank_ids = tokenizer.blank_ids
        for i in range(MAX_BLANKS):
            assert static_ids[i] == actual_blank_ids[i]
