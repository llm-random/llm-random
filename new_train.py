import random

from datasets import load_dataset
from transformers import BertTokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset


# dataset_book = load_dataset("bookcorpus")
# dataset_wiki = load_dataset("wikipedia", "20220301.en")
#
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
#
# print(tokenizer.encode("Hello, my name is John.", add_special_tokens=True))
#
#
# print(dataset_wiki)


class SentencePair(object):
    def __init__(self, sen1, sen2):
        self.sen1 = sen1
        self.sen2 = sen2


class SentencePairProcessor(object):
    def __init__(self, max_total_length=128):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_total_length = max_total_length
        self.max_sentence_length = (max_total_length - 4) // 2

    def process(self, sentence_pair):
        sen1 = self.tokenizer.encode(sentence_pair.sen1)
        if len(sen1) > self.max_sentence_length:
            sen1 = sen1[-self.max_sentence_length:]
        sen2 = self.tokenizer.encode(sentence_pair.sen2)
        if len(sen2) > self.max_sentence_length:
            sen2 = sen2[:self.max_sentence_length]
        together = self.tokenizer.build_inputs_with_special_tokens(sen1, sen2)
        return together


class WikipediaDataset(object):
    def __init__(self, evaluate: bool = False):
        self.examples_buffer = []
        self.dataset_wiki = load_dataset("wikipedia", "20220301.en")['train']
        self.dataset_book = load_dataset("bookcorpus")['train']

        self.buffer_refill_to = 10000
        self.buffer_refill_from = 0
        self.min_sentence_length = 40
        self.bookcorpus_chance = 0.5
        self.bookcorpus_lines = len(self.dataset_book)//len(self.dataset_wiki)+1
        self.bookcorpus_chance = self.bookcorpus_chance / 100 * self.bookcorpus_lines
        self.bookcorpus_lines = 100 # the above is very approximate
        self.wikipedia_chance = 1.0 - self.bookcorpus_chance
        print("bookcorpus_lines:", self.bookcorpus_lines)
        print("bookcorpus_chance:", self.bookcorpus_chance)

    def get_example(self):
        if len(self.examples_buffer) <= self.buffer_refill_from:
            self._refill_buffer()
        example = self.examples_buffer.pop()
        return example

    def _refill_buffer(self):
        while len(self.examples_buffer) <= self.buffer_refill_to:
            last_len = len(self.examples_buffer)
            self._add_examples(self._get_random_document())
        random.shuffle(self.examples_buffer)

    def _get_random_document(self):
        if random.random() < self.wikipedia_chance:
            document_text = self.dataset_wiki[random.randint(0, len(self.dataset_wiki) - 1)]['text']
            document_sentences = document_text.replace('.', '\n').split('\n')
            assert isinstance(document_sentences, list)
            assert isinstance(document_sentences[0], str)
        else:
            linebegin = random.randint(0, len(self.dataset_wiki) - 1 - self.bookcorpus_lines)
            lineend = linebegin + self.bookcorpus_lines
            document_sentences = self.dataset_book[linebegin:lineend]
            document_sentences = [sentence for sentence in document_sentences]
            assert isinstance(document_sentences, list)
            assert isinstance(document_sentences[0], str)
        return document_sentences

    def _add_examples(self, document_sentences):
        emptysentencelength = 5
        document_sentences.append("a"*emptysentencelength)  # hack to ensure last sentences can be added
        good_sentences = []
        for sentence in document_sentences:
            if len(sentence) > self.min_sentence_length:
                good_sentences.append(sentence)
            elif len(sentence.strip()) < emptysentencelength:
                continue
            else:
                if len(good_sentences) % 2 == 1:
                    if random.random() < 0.5:
                        good_sentences.pop()
                    else:
                        good_sentences.pop(0)
                for i in range(0, len(good_sentences), 2):
                    pair = SentencePair(good_sentences[i], good_sentences[i + 1])
                    self.examples_buffer.append(pair)
                good_sentences = []

dat = WikipediaDataset()
dat.get_example().encode()
