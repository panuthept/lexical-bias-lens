import os
import json
import math
from copy import deepcopy
from model import LexicalBiasModel
from collections import defaultdict
from utils.ngrams import get_ngrams
from typing import Literal, List, Tuple, Any, Optional


def calculate_normalized_entropy(pred, num_classes) -> float:
    return sum([-math.log(p) * p if p > 0 else 0 for c, p in pred]) / math.log(num_classes)

class LexicalBiasLens(LexicalBiasModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = []
        self.preds = []
        self.entropies = []

    def clear(self):
        super().clear()
        self.labels = []
        self.preds = []
        self.entropies = []

    def create_profile(self, inputs: List[List[Any]], labels: List[str], verbose: bool = True) -> None:
        """
        Profiles the dataset and computes the entropy scores for each sample.
        """
        self.clear()
        self.labels = deepcopy(labels)
        self.fit(inputs, labels, verbose=verbose)
        self.preds = self.predict(inputs, verbose=verbose)
        self.entropies = [calculate_normalized_entropy(pred, num_classes=self.num_classes) for pred in self.preds]

    def find(
        self, 
        target_label: str = None,
        target_pred: str = None,
        max_entropy: float = 1.0,
        min_entropy: float = 0.0,
        ranking_order: Optional[Literal["ascending", "decending"]] = None, 
        inputs: List[List[Any]] = None, 
        labels: List[str] = None, 
        verbose: bool = True,
    ) -> List[int]:
        """
        Identifies and returns the indices of the samples that meet the specified criteria.
        """
        if target_label is not None:
            assert target_label in self.available_labels, f"target_label '{target_label}' not found in available labels: {self.available_labels}"
        if target_pred is not None:
            assert target_pred in self.available_labels, f"target_pred '{target_pred}' not found in available labels: {self.available_labels}"
        assert 0.0 <= min_entropy <= max_entropy <= 1.0, "Entropy bounds must satisfy 0.0 <= min_entropy <= max_entropy <= 1.0"
        assert ranking_order in [None, "ascending", "decending"], "ranking_order must be one of None, 'ascending', or 'decending'"

        if inputs is not None and labels is not None:
            self.create_profile(inputs, labels, verbose=verbose)

        filtered_indices = []
        for i, (label, pred, entropy) in enumerate(zip(self.labels, self.preds, self.entropies)):
            pred_label = pred[0][0]  # Get the label with the highest probability
            if (target_label is None or label == target_label) and \
               (target_pred is None or pred_label == target_pred) and \
               (entropy <= max_entropy) and \
               (entropy >= min_entropy):
                filtered_indices.append(i)
        if ranking_order == "ascending":
            filtered_indices = sorted(filtered_indices, key=lambda i: self.entropies[i])
        elif ranking_order == "decending":
            filtered_indices = sorted(filtered_indices, key=lambda i: self.entropies[i], reverse=True)
        return filtered_indices
    
    def analyze(self, tokens: List[Any], metric: str = None) -> List[Tuple[str, float]]:
        """
        Analyzes a single input sample and returns its bias profile scores.
        """
        assert self.bias_profile is not None, "Please fit() the model before analysis."
        if metric is None:
            metric = self.metric

        token_cnts = defaultdict(int)
        token_bias = defaultdict(lambda: defaultdict(float))
        for ngram in get_ngrams(tokens, min_n=self.min_n, max_n=self.max_n):
            token_cnts[ngram] += 1
            for label in self.bias_profile:        
                if ngram in self.bias_profile[label]:
                    if self.bias_profile[label][ngram][metric] >= 0:
                        token_bias[label][ngram] = self.bias_profile[label][ngram][metric]
                else:
                    token_bias[label][ngram] = 0.0

        results = {label: [(" ".join(token), token_cnts[token], score) for token, score in token_scores.items()] for label, token_scores in token_bias.items()}
        results = {label: sorted(analysis, key=lambda x: x[2], reverse=True) for label, analysis in results.items()}
        return results
    
    def save(self, filepath: str) -> None:
        super().save(filepath)
        data_profile = {
            "labels": self.labels,
            "preds": self.preds,
            "entropies": self.entropies
        }
        with open(os.path.join(filepath, "data_profile.json"), "w") as f:
            json.dump(data_profile, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, filepath: str, metric: str = None) -> 'LexicalBiasLens':
        model = super().load(filepath, metric=metric)
        with open(os.path.join(filepath, "data_profile.json"), "r") as f:
            data_profile = json.load(f)
        lens = cls(
            max_n=model.max_n,
            min_n=model.min_n,
            min_freq=model.min_freq,
            metric=model.metric,
            epsilon=model.epsilon,
        )
        lens.data_stats = model.data_stats
        lens.build_bias_profile()
        lens.labels = data_profile["labels"]
        lens.preds = data_profile["preds"]
        lens.entropies = data_profile["entropies"]
        return lens
    

if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("aisingapore/SEA-Safeguard-Train-Cultural-v3", "Generic", split="train")
    
    inputs = []
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
        inputs.append(input)
        labels.append(label)

    if not os.path.exists("saved_models/lexical_bias_lens"):
        lens = LexicalBiasLens(max_n=1, metric="LMI")
        lens.create_profile(inputs, labels)
        lens.save("saved_models/lexical_bias_lens")
    else:
        lens = LexicalBiasLens.load("saved_models/lexical_bias_lens", metric="LMI")
    print(f"Average Entropy: {sum(lens.entropies) / len(lens.entropies)}")
    indices = lens.find(target_label="Unsafe", target_pred="Unsafe", ranking_order="decending")
    for i in indices[:10]:
        print(f"Input: {" ".join(inputs[i])}")
        print(f"Label: {labels[i]}")
        print(f"Predictions: {lens.preds[i]}")
        print(f"Entropy: {lens.entropies[i]}")
        print("Analyzed:")
        for label, analysis in lens.analyze(inputs[i]).items():
            print(f"  Label: {label}")
            sum_score = sum([score * count for token, count, score in analysis])
            print(f"    Total Score: {sum_score}")
            for token, count, score in analysis[:5]:
                print(f"    Token: {token}, Count: {count}, Score: {score}")
        print("-----")
    print("...")
    indices = lens.find(target_label="Unsafe", target_pred="Unsafe", ranking_order="ascending")
    for i in indices[:10]:
        print(f"Input: {" ".join(inputs[i])}")
        print(f"Label: {labels[i]}")
        print(f"Predictions: {lens.preds[i]}")
        print(f"Entropy: {lens.entropies[i]}")
        print("Analyzed:")
        for label, analysis in lens.analyze(inputs[i]).items():
            print(f"  Label: {label}")
            sum_score = sum([score * count for token, count, score in analysis])
            print(f"    Total Score: {sum_score}")
            for token, count, score in analysis[:5]:
                print(f"    Token: {token}, Count: {count}, Score: {score}")
        print("-----")