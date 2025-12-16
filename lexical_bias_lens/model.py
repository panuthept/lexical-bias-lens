import os
import json
import math
from tqdm import tqdm
from collections import defaultdict
from typing import List, Set, Dict, Any
from .utils import get_ngrams, hash_sample


class LexicalBiasModel:
    def __init__(
        self, 
        metric: str = "LMI", 
        min_freq: int = 1,  # Increasing this will reduce vocab size by removing rare vocabs, save memory. However, PMI and NPMI metrics benefit from rare vocabs.
        min_n: int = 1, 
        max_n: int = 3,     # Increasing this might increase performance but also increase vocab size, computation time, and memory usage.
        epsilon: float = 1e-10,
    ):
        self.metric = metric
        self.min_freq = min_freq
        self.min_n = min_n
        self.max_n = max_n
        self.epsilon = epsilon

        self.sample_hashs: Set[str] = set()
        self.data_stats: Dict[str, Dict[tuple, int]] = None
        self.bias_profile: Dict[str, Dict[tuple, float]] = None

    @property
    def vocab_size(self) -> int:
        if self.data_stats is None:
            return 0
        vocab = set()
        for label_stats in self.data_stats.values():
            vocab.update(label_stats.keys())
        return len(vocab)
    
    @property
    def available_labels(self) -> List[str]:
        if self.data_stats is None:
            return []
        return list(self.data_stats.keys())
    
    @property
    def num_classes(self) -> int:
        if self.data_stats is None:
            return 0
        return len(self.data_stats)
    
    def clear(self) -> None:
        self.sample_hashs = set()
        self.data_stats = None
        self.bias_profile = None

    def fit(self, samples: List[List[Any]], labels: List[str], verbose: bool = True) -> None:
        if self.data_stats is None:
            self.data_stats = defaultdict(lambda: defaultdict(int)) # count(W, Y)

        # Get new_samples by removing already seen samples
        new_samples = []
        new_labels = []
        for tokens, label in zip(samples, labels):
            sample_hash = hash_sample(tokens)
            if sample_hash in self.sample_hashs:
                continue
            new_samples.append(tokens)
            new_labels.append(label)

        if len(new_samples) == 0:
            return

        with tqdm(total=len(new_samples), disable=not verbose, desc="Fitting new samples") as pbar:
            for tokens, label in zip(new_samples, new_labels):
                for ngram in get_ngrams(tokens, min_n=self.min_n, max_n=self.max_n):
                    self.data_stats[label][ngram] += 1

                sample_hash = hash_sample(tokens)
                self.sample_hashs.add(sample_hash)

                pbar.update(1)

    def build_bias_profile(self) -> None:
        assert self.data_stats is not None, "Please fit() the model before building bias profile."

        # Filter n-grams by min_freq
        filtered_data_stats = {}
        for label in self.data_stats:
            filtered_data_stats[label] = {ngram: freq for ngram, freq in self.data_stats[label].items() if freq >= self.min_freq}

        total_ngarms = 0
        ngram_freq = defaultdict(int) # count(W)
        label_ngram_freq = defaultdict(lambda: defaultdict(int)) # count(W, Y)
        for label in filtered_data_stats:
            for ngram, freq in filtered_data_stats[label].items():
                total_ngarms += freq
                ngram_freq[ngram] += freq
                label_ngram_freq[label][ngram] = freq

        self.bias_profile = {}
        for label in filtered_data_stats:

            label_stats = {}

            total_ngrams_in_label = sum(label_ngram_freq[label].values())
            for ngram, count_WY in label_ngram_freq[label].items():
                p_W = ngram_freq[ngram] / total_ngarms  # P(W)
                p_WY = count_WY / total_ngarms  # P(W,Y)
                p_W_given_Y = count_WY / total_ngrams_in_label # P(W|Y)

                PMI = math.log((p_W_given_Y + self.epsilon) / (p_W + self.epsilon))
                NPMI = PMI / (-math.log(p_WY + self.epsilon))

                LMI = count_WY * PMI # freq × PMI

                label_stats[ngram] = {"LMI": LMI, "PMI": PMI, "NPMI": NPMI}
            self.bias_profile[label] = label_stats

    def predict(self, samples: List[List[str]], verbose: bool = True) -> List[Dict[str, float]]:
        if self.bias_profile is None and self.data_stats is not None:
            self.build_bias_profile()
        assert self.bias_profile is not None, "Please fit() the model before prediction."
        preds = []
        for tokens in tqdm(samples, disable=not verbose, desc="Predicting"):
            class_scores = {label: 0.0 for label in self.bias_profile.keys()}
            for label in class_scores:
                stats = self.bias_profile[label]
                for w in get_ngrams(tokens, min_n=self.min_n, max_n=self.max_n):
                    class_scores[label] += max(stats.get(w, {self.metric: 0.0})[self.metric], 0.0)

            if sum(class_scores.values()) == 0:
                class_probs = {label: 1.0 / len(class_scores) for label in class_scores}
            else:
                class_probs = {label: score / sum(class_scores.values()) for label, score in class_scores.items()}
            preds.append(sorted([(label, class_probs[label], class_scores[label]) for label in class_probs.keys()], key=lambda x: x[1], reverse=True))
        return preds
    
    def save(self, filepath: str) -> None:
        serializable_stats = {
            label: {str(ngram): value for ngram, value in ngram_stats.items()}
            for label, ngram_stats in self.data_stats.items()
        }
        model_params = {
            "max_n": self.max_n,
            "min_n": self.min_n,
            "min_freq": self.min_freq,
            "metric": self.metric,
            "epsilon": self.epsilon,
            "data_stats": serializable_stats,
        }
        os.makedirs(filepath, exist_ok=True)
        with open(os.path.join(filepath, "model_params.json"), "w") as f:
            json.dump(model_params, f, ensure_ascii=False, indent=4)
        with open(os.path.join(filepath, "sample_hashs.txt"), "w") as f:
            f.write('\n'.join(list(self.sample_hashs)))

    @classmethod
    def load(cls, filepath: str, metric: str = None) -> 'LexicalBiasModel':
        assert os.path.exists(os.path.join(filepath, "model_params.json")), f"Model file not found at {os.path.join(filepath, 'model_params.json')}"
        with open(os.path.join(filepath, "model_params.json"), "r") as f:
            model_params = json.load(f)
        with open(os.path.join(filepath, "sample_hashs.txt"), "r") as f:
            sample_hashs = f.read().splitlines()
        model_params["sample_hashs"] = set(sample_hashs)
        if metric is not None:
            assert metric in ["LMI", "PMI", "NPMI"], "Metric must be one of ['LMI', 'PMI', 'NPMI']"
            model_params["metric"] = metric

        deserialized_stats = {
            label: {eval(ngram): value for ngram, value in ngram_stats.items()}
            for label, ngram_stats in model_params.pop("data_stats").items()
        }

        # Remove unknown parameters for backward compatibility
        known_params = {"max_n", "min_n", "min_freq", "metric", "epsilon"}
        model_params = {k: v for k, v in model_params.items() if k in known_params}
        model = cls(**model_params)
        model.data_stats = deserialized_stats
        model.sample_hashs = set(model_params.get("sample_hashs", []))
        model.build_bias_profile()
        return model
    

if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("aisingapore/SEA-Safeguard-Train-Cultural-v3", "Generic", split="train")
    
    samples = []
    labels = []
    for sample in dataset:
        if sample['language'] != "English":
            continue
        if sample["response"] is None:
            input = sample['prompt'].split()
            label = sample['prompt_label']
        else:
            input = (sample['prompt'] + "\n" + sample['response']).split()
            label = sample['response_label']
        samples.append(input)
        labels.append(label)

    save_path = "saved_models/lexical_bias_model"
    if not os.path.exists(save_path):
        model = LexicalBiasModel(max_n=1, metric="NLMI")
        model.fit(samples, labels)
        model.save(save_path)
    else:
        model = LexicalBiasModel.load(save_path, metric="NLMI")
    print("Vocab size:", model.vocab_size)
    outputs = model.predict(samples)

    tp = 0
    fp = 0
    fn = 0
    for label, output in zip(labels, outputs):
        if label == "Unsafe":
            if output[0][0] == "Unsafe":
                tp += 1
            else:
                fn += 1
        else:
            if output[0][0] == "Unsafe":
                fp += 1
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp) / (tp + fp + fn + 1e-8)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Accuracy:", accuracy)
