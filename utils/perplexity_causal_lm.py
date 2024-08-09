import evaluate
import numpy as np


def get_perplexities(model_id: str, input_texts: list[str]):
    perplexity = evaluate.load("perplexity", module_type="metric")


if __name__ == "__main__":
    model_id = "gpt2"

    perplexity = evaluate.load("perplexity", module_type="metric")
    input_texts = ["lorem ipsum", "Happy Birthday!", "hello how are you"]

    results: dict[str, list[float] | np.float64] = perplexity.compute(
        model_id=model_id, add_start_token=True, predictions=input_texts
    )
    print(results)
