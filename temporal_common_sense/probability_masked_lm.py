import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer

model_name = "roberta-base"
model = RobertaForMaskedLM.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Input sentence with a masked token <mask>
input_sentences = [
    f"I enjoy {tokenizer.mask_token} in the park.",
]

candidate_list = ["walking", "reading"]

# Tokenize and encode the input sentence
inputs = tokenizer(input_sentences, return_tensors="pt", padding=True, truncation=True)

# Get logits for the masked token
logits = model(**inputs).logits     
# Dimensions: (sentence, token_index_in_sentence, token_in_vocab)
# Shape: (num_sentences, seq_length, vocab_size)

for i in range(len(input_sentences)):
    # Extract probability distribution for the masked tokens in the current sentence
    masked_token_indices = torch.where(inputs["input_ids"][i] == tokenizer.mask_token_id)[0]
    probabilities = logits[i, masked_token_indices].softmax(dim=1)
    # Dimensions: (sentence: int, token: str)
    # Shape: (num_sentences, vocab_size)

    # Display the probabilities for each candidate
    print(f"Probabilities for sentence {i}:")
    for candidate in candidate_list:
        candidate_index = tokenizer.convert_tokens_to_ids(candidate)
        probability_of_candidate = probabilities[:, candidate_index].item()
        print(f"  Probability of '{candidate}': {probability_of_candidate}")
    print()

