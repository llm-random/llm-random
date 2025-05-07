from data.preprocess import preprocess_data
# import json
# import os
# from tqdm import tqdm
# from datasets import load_from_disk

if __name__ == "__main__":
    print("Starting preprocessing")

    
    # dataset = load_from_disk("/nemo_run/datasets/c4/train")
    
    # total_samples = len(dataset)
    # i = 0
    # parts = 10
    # samples_per_part = (total_samples // parts) + 1

    # os.makedirs("c4_splits", exist_ok=True)
    
    # for i in range(parts):
    #     start = i * samples_per_part
    #     end = start + samples_per_part if i < parts - 1 else total_samples
    #     with open(f"c4_splits/c4_en_train_part_{i:02}.jsonl", "w", encoding="utf-8") as f:
    #         for ex in tqdm(dataset.select(range(start, end)), desc=f"Part {i}", total=end - start):
    #             json.dump({"text": ex["text"]}, f)
    #             f.write("\n")


    # preprocess_data(
    #     # data_dir="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/c4_concat_dev/", 
    #     data_dir="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/c4_concat/", 
    #     output_dir="./preprocessing_results", 
    #     # tokenizer_model="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/llama2/tokenizer.model",
    #     tokenizer_library="huggingface",
    #     vocab_file_path="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/gpt2/vocab.json",
    #     merges_file_path="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/gpt2/merges.txt",
    # )

    preprocess_data(
        # data_dir="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/c4_concat_dev/", 
        # data_dir="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/c4_concat/", 
        data_dir="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/c4_concat/",  # c4_en_train_part_00.jsonl
        output_dir="./results_preprocessing", 
        # tokenizer_model="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/llama2/tokenizer.model",
        # tokenizer_library="megatron",
        tokenizer_library="huggingface",
        # vocab_file_path="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/gpt2/vocab.json",
        # merges_file_path="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/gpt2/merges.txt",
        # tokenizer_type="GPT2BPETokenizer",
        tokenizer_type="gpt2",
    )


    print("Finished preprocessing")


    # python scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    # --input=PATH_TO_THE_RETRIEVAL_DB_LOOSE_JSON_FILE \
    # --json-keys=text \
    # --tokenizer-library=megatron \
    # --tokenizer-type=GPT2BPETokenizer \
    # --dataset-impl=mmap \
    # --merge-file=YOUR_MERGE_FILE \
    # --vocab-file=YOUR_VOCAB_FILE \
    # --output-prefix=YOUR_DATA_PREFIX \
    # --append-eod \
    # --workers=48


    # data_dir="/data/slimpajama",
    # output_dir="/data/slimpajama_megatron",
    # tokenizer_model="/data/tokenizer/tokenizer.model",
    # tokenizer_library="sentencepiece",

    # /net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers
    # 
    # 