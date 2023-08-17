from datasets import load_dataset
import tqdm
import plotly.express as px
from transformers import GPT2Tokenizer
import random

d = load_dataset("c4", "en", split="train")
print(f"Number of examples: {len(d)}")
print(f"First example: {d[0]}")

t = GPT2Tokenizer.from_pretrained("gpt2")

lengths = []
for _ in tqdm.tqdm(range(100_000)):
    row_num = random.randint(0, len(d))
    example = d[row_num]
    lengths.append(len(t(example["text"])["input_ids"]))

fig = px.histogram(lengths)
# save figure to file as html
print(f"Total number of tokens in these examples: {sum(lengths)}")
print(f"Total number of rows in dataset: {len(d)}")
print(f"Estimated number of tokens in dataset: {sum(lengths) * len(d) / len(lengths)}")
fig.write_html("c4_lengths.html")
