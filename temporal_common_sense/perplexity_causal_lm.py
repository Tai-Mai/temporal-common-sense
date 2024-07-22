import evaluate

model = "gpt2"

perplexity = evaluate.load("perplexity", module_type="metric")
input_texts = ["lorem ipsum", "Happy Birthday!"]

results = perplexity.compute(
    model_id=model, add_start_token=True, predictions=input_texts
)
print(results)
