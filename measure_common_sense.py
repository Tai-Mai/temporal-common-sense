import evaluate
import fire
from functools import partial
import json
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
from typing import Callable

from utils.metrics import Metric, Perplexity, PseudoPerplexity


type Relation = str
type Verbalization = str
type Event = str
type Count = int


def main(lm_mode: str = "causal", model_id=None):
    data_filepath = "data/claude_examples.json"

    match lm_mode:
        case "causal":
            metric: Metric = Perplexity(model_id=model_id or "gpt2")
        case "masked":
            metric: Metric = PseudoPerplexity(model_id=model_id or "roberta-base")
        case _:
            raise ValueError("Invalid language modeling mode")

    with open(data_filepath) as f:
        data: dict[
            Relation, dict[str, list[Verbalization] | list[dict[str, Event]]]
        ] = json.load(f)

    confusion_matrices: dict[Relation, list[dict[Relation, Count]]] = {}

    true_relation_pbar = tqdm(data.items(), leave=False)
    for true_relation, true_relation_data in true_relation_pbar:
        true_relation_pbar.set_description(f"True relation {true_relation}")
        # confusion matrix to count how often a certain relation scored best for a true relation
        confusions: dict[Relation, Count] = {relation: 0 for relation in data}

        for example in tqdm(
            true_relation_data["examples"], desc="Examples", leave=False
        ):
            event1, event2 = example["event1"], example["event2"]
            # for the current event pair example, record the metric values of each possible relation
            relation_metric_values: dict[Relation, float] = {}
            relation_pbar = tqdm(data.items(), leave=False)
            for relation, relation_data in relation_pbar:
                relation_pbar.set_description(f"Comparing relation {relation}")
                verbalizations = [
                    verbalization.format(event1=event1, event2=event2)
                    for verbalization in relation_data["verbalizations"]
                ]
                metric_values: dict[str, list[float] | float] = metric(verbalizations)
                relation_metric_values[relation] = metric_values["average"]
            predicted_relation: Relation = max(
                relation_metric_values, key=relation_metric_values.get
            )
            confusions[predicted_relation] += 1
        confusion_matrices[true_relation] = confusions
    print(confusion_matrices)


if __name__ == "__main__":
    fire.Fire(main)
