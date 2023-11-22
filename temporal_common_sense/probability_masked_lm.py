import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer

model_name = "roberta-base"
model = RobertaForMaskedLM.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Input sentence with a masked token <mask>
input_sentences = [
    f"I like {tokenizer.mask_token}.",
]

candidate_list = ["question", "them both"]

inputs = tokenizer(input_sentences, return_tensors="pt", padding=True, truncation=True)

logits = model(**inputs).logits     
# Dimensions: (sentence, token_index_in_sentence, token_in_vocab)
# Shape: (num_sentences, seq_length, vocab_size)

for i, sentence in enumerate(input_sentences):
    # Position of the mask token in the input sentences
    masked_token_indices = torch.where(inputs["input_ids"][i] == tokenizer.mask_token_id)[0]
    probabilities = logits[i, masked_token_indices].softmax(dim=1)
    # Dimensions: (sentence: int, token: str)
    # Shape: (num_sentences, vocab_size)

    print(f'Probabilities for sentence "{sentence}":')
    for candidate in candidate_list:
        candidate_index = tokenizer.convert_tokens_to_ids(candidate)
        probability_of_candidate = probabilities[:, candidate_index].item()
        print(f"  Probability of '{candidate}': {probability_of_candidate}")
    print()

