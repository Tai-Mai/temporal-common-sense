from abc import ABC
from copy import deepcopy
import evaluate
import numpy as np
from math import log, exp
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import torch


class Metric(ABC):
    def __init__(self, model: PreTrainedModel) -> None:
        raise NotImplementedError

    def __call__(self, sentences: list[str]) -> dict[str, list[float] | float]:
        raise NotImplementedError


class Perplexity(Metric):
    """
    Returns the pseudo perplexity for a list of sentences. Only designed for causal language models
    """

    def __init__(self, model: PreTrainedModel) -> None:
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model: PreTrainedModel = model
        self.premade_metric = evaluate.load("perplexity", module_type="metric")
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.model.name_or_path
        )

    def _use_premade_metric(
        self, sentences: list[str]
    ) -> dict[str, list[float] | float]:
        """
        DEPRECATED
        Returns the perplexities of each sentence as well as the mean perplexity across all sentences using the huggingface metric. This seems to return inflated scores.

        :param sentences: List of sentences to measure perplexity for
        """
        results: dict[str, list[float] | np.float64] = self.premade_metric.compute(
            model_id=self.model_id, add_start_token=True, predictions=sentences
        )

        return {
            "values": results["perplexities"],
            "average": float(results["mean_perplexity"]),
        }

    def __call__(self, sentences: list[str]) -> dict[str, list[float] | float]:
        """
        Returns the perplexities of each sentence as well as the mean perplexity across all sentences using the model outputs. This seems to return more sensible scores than the huggingface metric.

        :param sentences: List of sentences to measure perplexity for
        :returns: single perplexity values for each sentence, as well as average perplexity
        """
        log_likelihoods: list[float] = []
        perplexities: list[float] = []
        for sentence in sentences:
            tokenized_sentence: torch.Tensor = self.tokenizer.encode(
                sentence, return_tensors="pt"
            )
            with torch.no_grad():
                output = self.model(
                    input_ids=tokenized_sentence.to(self.model.device), 
                    labels=tokenized_sentence.clone().to(self.model.device),
                )
                log_likelihood: torch.Tensor = output.loss.detach()
                log_likelihoods.append(log_likelihood.item())
                perplexity = torch.exp(output.loss.detach()).item()
                perplexities.append(perplexity)
        return {
            "values": perplexities,
            "average": sum(perplexities) / len(perplexities),
        }


class PseudoPerplexity(Metric):
    """
    Returns the pseudo perplexity for a list of sentences. Only designed for masked language models.
    """

    def __init__(self, model: PreTrainedModel):
        self.model: PreTrainedModel = model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.name_or_path)

    def __call__(self, sentences: list[str]) -> dict[str, list[float] | float]:
        """
        Returns the perplexities of each sentence as well as the mean perplexity across all sentences using the model outputs. This seems to return more sensible scores than the huggingface metric.

        :param sentences: List of sentences to measure pseudo-perplexity for
        :returns: single pseudo-perplexity values for each sentence, as well as average pseudo-perplexity
        """
        assert len(sentences) > 0

        # pseudo_log_likelihoods: list[float] = []
        pseudo_perplexities: list[float] = []
        num_total_tokens: int = 0
        for sentence in sentences:
            tokenized_sentence: torch.Tensor = self.tokenizer.encode(
                sentence, return_tensors="pt"
            )
            num_tokens = tokenized_sentence.shape[-1]
            num_total_tokens += num_tokens

            pseudo_log_likelihood = self.pseudo_log_likelihood(tokenized_sentence)
            # pseudo_log_likelihoods.append(pseudo_log_likelihood)

            pseudo_perplexity = exp(-1 / num_tokens * pseudo_log_likelihood)
            pseudo_perplexities.append(pseudo_perplexity)

        # average_pseudo_perplexity: float = exp(
        #     -1 / num_total_tokens * sum(pseudo_log_likelihoods)
        # )
        average_pseudo_perplexity: float = sum(pseudo_perplexities) / len(
            pseudo_perplexities
        )
        return {"values": pseudo_perplexities, "average": average_pseudo_perplexity}

    def pseudo_log_likelihood(
        self,
        tokenized_sentence: torch.Tensor,
    ) -> float:
        """
        Calculates the pseudo-log-likelihood (PLL) for a sentence under a model and tokenizer by masking every token in the sentence, one by one, and adding up all log probabilities of the masked token appearing at its position.
        """
        pseudo_log_likelihood = 0
        for token_position, original_token_id in enumerate(
            tokenized_sentence.squeeze()
        ):
            masked_sentence = tokenized_sentence.clone()
            masked_sentence[:, token_position] = self.tokenizer.mask_token_id
            with torch.no_grad():
                output = self.model(input_ids=masked_sentence.to(self.model.device))
                logits: torch.Tensor = output.logits.squeeze()
            probabilities = logits[token_position].softmax(dim=0)
            probability = probabilities[original_token_id]
            # top = torch.max(probabilities, dim=0, keepdim=True)
            pseudo_log_likelihood += log(probability)

        return pseudo_log_likelihood


if __name__ == "__main__":
    perplexity_metric = Perplexity(model_id="gpt2")
    pseudo_perplexity_metric = PseudoPerplexity(model_id="roberta-base")
    test_sentences = ["how are you", "how are you doing", "abcde", "abcdefghijk"]
    print(f"{perplexity_metric(test_sentences)=}")
    print(f"{perplexity_metric.use_premade_metric(test_sentences)=}")
    print(f"{pseudo_perplexity_metric(test_sentences)=}")
