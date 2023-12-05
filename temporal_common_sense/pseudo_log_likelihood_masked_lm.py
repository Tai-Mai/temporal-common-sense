from math import log
from copy import deepcopy
from icecream import ic
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer


def calculate_pseudo_log_likelihood(sentence: str, 
                                    model: RobertaForMaskedLM,
                                    tokenizer: RobertaTokenizer) -> float:
    """
    Calculates the pseudo-log-likelihood (PLL) for a sentence under a model and tokenizer by masking every token in the sentence, one by one, and adding up all log probabilities of the masked token appearing at its position.
    """

    tokenized_sentence = tokenizer(sentence, return_tensors="pt")
    original_token_ids = deepcopy(tokenized_sentence.input_ids[0])

    pseudo_log_likelihood = 0
    for token_position, original_token_id in enumerate(original_token_ids):
        tokenized_sentence.input_ids[0, token_position] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(**tokenized_sentence).logits[0]
        probabilities = logits[token_position].softmax(dim=0)
        probability = probabilities[original_token_id]
        top = torch.max(probabilities, dim=0, keepdim=True)
        pseudo_log_likelihood += log(probability)

    return pseudo_log_likelihood


def main():
    model_name = "roberta-base"
    model = RobertaForMaskedLM.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    input_sentences = [
        "I enjoy eating in the restaurant.",
        "I enjoy xxxxxxxxxxxxxxx in the restaurant.",
    ]

    for sentence in input_sentences:
        pseudo_log_likelihood = calculate_pseudo_log_likelihood(sentence, model, tokenizer)
        ic(pseudo_log_likelihood)


if __name__ == "__main__":
    main()
