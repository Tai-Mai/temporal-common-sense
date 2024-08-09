from math import log, exp
from copy import deepcopy
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def pseudo_perplexity(
    sentence: str,
    model,
    tokenizer,
) -> float:
    """
    Calculates the pseudo-log-likelihood (PLL) for a sentence under a model and tokenizer by masking every token in the sentence, one by one, and adding up all log probabilities of the masked token appearing at its position.
    """

    tokenized_sentence = tokenizer(sentence, return_tensors="pt")
    original_token_ids = deepcopy(tokenized_sentence.input_ids.squeeze())

    pseudo_log_likelihood = 0
    for token_position, original_token_id in enumerate(original_token_ids):
        tokenized_sentence.input_ids[0, token_position] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(**tokenized_sentence).logits.squeeze()
        probabilities = logits[token_position].softmax(dim=0)
        probability = probabilities[original_token_id]
        # top = torch.max(probabilities, dim=0, keepdim=True)
        pseudo_log_likelihood += log(probability)

    return exp(-1 / len(original_token_ids) * pseudo_log_likelihood)


def main():
    model_id = "roberta-base"
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    input_sentences = [
        "I enjoy eating in the restaurant because the food there tastes great.",
        "I enjoy xxxxxxxxxxxxxxx in the restaurant.",
    ]

    for sentence in input_sentences:
        pppl = pseudo_perplexity(sentence, model, tokenizer)
        print(f"{pppl=}")


if __name__ == "__main__":
    main()
