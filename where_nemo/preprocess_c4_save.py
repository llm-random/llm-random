from data.preprocess import preprocess_data

if __name__ == "__main__":

    # dataset = load_from_disk("/nemo_run/datasets/c4/train")
    
    # # Optional: get the total number of samples for tqdm (only if not streaming)
    # total_samples = len(dataset)
    # i = 0

    # # Save to JSONL with progress bar
    # with open("c4_en_train.jsonl", "w", encoding="utf-8") as f:
    #     for example in tqdm(dataset, total=total_samples, desc="Converting to JSONL"):
    #         if i > 20_000_000:
    #             break
    #         json.dump({"text": example["text"]}, f)
    #         f.write("\n")
    #         i+=1
    print("Starting preprocessing")
    preprocess_data(
        data_dir="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/c4_concat_dev/", 
        output_dir="./preprocessing_results", 
        tokenizer_model="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/tokenizer.model",
        tokenizer_library="sentencepiece"
    )
    print("Finished preprocessing")

    # data_dir="/data/slimpajama",
    # output_dir="/data/slimpajama_megatron",
    # tokenizer_model="/data/tokenizer/tokenizer.model",
    # tokenizer_library="sentencepiece",

    # /net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers
    # 
    # 