import os
from datasets import load_dataset
from transformers import AutoTokenizer
from lexical_bias_lens import LexicalBiasLens


if __name__ == "__main__":
    save_path = "saves/aisingapore/1M_SEA-Guard_qwen3-8b_Non_Bias/cultural/train_refined_v1_qa_pass_only/english_only"
    tokenizer = AutoTokenizer.from_pretrained("aisingapore/1M_SEA-Guard_qwen3-8b_Non_Bias")

    dataset_sources = [
        # ("aisingapore/SEA-Safeguard-Train-Cultural-v3", "Indonesia", "train_refined_v1_qa_pass_only"),
        # ("aisingapore/SEA-Safeguard-Train-Cultural-v3", "Malaysia", "train_refined_v1_qa_pass_only"),
        # ("aisingapore/SEA-Safeguard-Train-Cultural-v3", "Myanmar", "train_refined_v1_qa_pass_only"),
        # ("aisingapore/SEA-Safeguard-Train-Cultural-v3", "Philippines", "train_refined_v1_qa_pass_only"),
        # ("aisingapore/SEA-Safeguard-Train-Cultural-v3", "Singapore", "train_refined_v1_qa_pass_only"),
        # ("aisingapore/SEA-Safeguard-Train-Cultural-v3", "Thailand", "train_refined_v1_qa_pass_only"),
        # ("aisingapore/SEA-Safeguard-Train-Cultural-v3", "Vietnam", "train_refined_v1_qa_pass_only"),
        # ("aisingapore/SEA-Safeguard-Train-Cultural-v3", "Generic", "train"),
    ]

    # if not os.path.exists(save_path):
    #     lens = LexicalBiasLens(max_n=1, metric="LMI")

    #     labels = []
    #     samples = []
    #     for repo, subset, split in dataset_sources:
    #         dataset = load_dataset(repo, subset, split=split)
        
    #         for sample in tqdm(dataset):
    #             if sample['language'] != "English":
    #                 continue
    #             if sample["response"] is None:
    #                 input = sample['prompt']
    #                 label = sample['prompt_label']
    #             else:
    #                 input = (sample['prompt'] + "\n" + sample['response'])
    #                 label = sample['response_label']
    #             tokens = tokenizer.tokenize(input)
    #             samples.append(tokens)
    #             labels.append(label)

    #             if len(samples) >= 10000:
    #                 lens.fit(samples, labels, verbose=False)
    #                 lens.save(save_path)
    #                 labels = []
    #                 samples = []
    # else:
    lens = LexicalBiasLens.load(save_path, metric="nLMI")

    keywords_scores = lens.find_bias_keywords(target_label="Unsafe", top_k=20, ranking_order="decending")
    keywords_scores = [(tokenizer.decode(tokenizer.convert_tokens_to_ids(token)), score) for token, score in keywords_scores]
    print(f"Top 20 bias keywords for 'Unsafe': {keywords_scores}")
    print()
    keywords_scores = lens.find_bias_keywords(target_label="Safe", top_k=20, ranking_order="decending")
    keywords_scores = [(tokenizer.decode(tokenizer.convert_tokens_to_ids(token)), score) for token, score in keywords_scores]
    print(f"Top 20 bias keywords for 'Safe': {keywords_scores}")