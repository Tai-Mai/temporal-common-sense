import fire
import json
from pathlib import Path
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    BitsAndBytesConfig,
    PreTrainedModel,
)
import torch
from tqdm import tqdm

from utils.metrics import Metric, Perplexity, PseudoPerplexity


Relation = str
Verbalization = str
Event = str
Count = int


def main(
    lm_mode: str = "causal",
    model_id: str = "",
    quantization: bool = False,
    normalize: bool = False,
):
    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # fp4 is default
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    if lm_mode == "causal":
        model_id = model_id or "gpt2"
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if quantization else None,
            cache_dir="./cache",
        )
        metric: Metric = Perplexity(model=model)
    elif lm_mode == "masked":
        model_id = model_id or "roberta-base"
        model: PreTrainedModel = AutoModelForMaskedLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if quantization else None,
            cache_dir="./cache",
        )
        metric: Metric = PseudoPerplexity(model=model)
    else:
        raise ValueError("Invalid language modeling mode")
    print(f"Model device: {model.device}")

    data_filepath: Path = Path("data/claude_examples.json")
    model_name: str = model_id.split("/")[-1].replace(".", "-").lower()

    confusion_matrix_dir: Path = Path("confusion_matrices")
    confusion_matrix_dir.mkdir(parents=True, exist_ok=True)

    confusion_matrix_counts_filepath: Path = (
        confusion_matrix_dir
        / f"{model_name}_confusion_matrix_counts{'_normalized' if normalize else ''}.json"
    )
    confusion_matrix_values_filepath: Path = (
        confusion_matrix_dir
        / f"{model_name}_confusion_matrix_values{'_normalized' if normalize else ''}.json"
    )

    with open(data_filepath) as f:
        data: dict[
            Relation, dict[str, list[Verbalization] | list[dict[str, Event]]]
        ] = json.load(f)

    confusion_matrix_counts: dict[Relation, dict[Relation, Count]] = {
        relation1: {relation2: 0 for relation2 in data} for relation1 in data
    }
    confusion_matrix_values: dict[Relation, dict[Relation, list[float]]] = {
        relation1: {relation2: [] for relation2 in data} for relation1 in data
    }

    if normalize:
        generic_verbalization_values: dict[Relation, list[float]] = {}
        generic_verbalization_averages: dict[Relation, float] = {}
        normalization_pbar = tqdm(data.items(), colour="yellow")
        for relation, relation_data in normalization_pbar:
            normalization_pbar.set_description(f'Normalizing "{relation}"')
            generic_verbalizations = [
                verbalization.replace("{event1}", "one event").replace(
                    "{event2}", "another event"
                )
                for verbalization in relation_data["verbalizations"]
            ]
            generic_verbalization_values[relation] = metric(generic_verbalizations)[
                "values"
            ]
            generic_verbalization_averages[relation] = metric(generic_verbalizations)[
                "average"
            ]

    true_relation_pbar = tqdm(data.items(), leave=False, colour="green")
    for true_relation, true_relation_data in true_relation_pbar:
        true_relation_pbar.set_description(f'True relation "{true_relation}"')
        # confusion matrix to count how often a certain relation scored best for a true relation

        for example in tqdm(
            true_relation_data["examples"], desc="Examples", leave=False, colour="blue"
        ):
            event1, event2 = example["event1"], example["event2"]
            # for the current event pair example, record the metric values of each possible relation
            relation_pbar = tqdm(data.items(), leave=False, colour="yellow")

            verbalized_relation_metric_averages: dict[Relation, float] = {}
            for verbalized_relation, relation_data in relation_pbar:
                relation_pbar.set_description(
                    f'Verbalized relation "{verbalized_relation}"'
                )
                verbalizations = [
                    verbalization.format(event1=event1, event2=event2)
                    for verbalization in relation_data["verbalizations"]
                ]
                results: dict[str, list[float] | float] = metric(verbalizations)
                values = results["values"]
                average = results["average"]

                if normalize:
                    values = [
                        value / generic_value
                        for value, generic_value in zip(
                            values, generic_verbalization_values[verbalized_relation]
                        )
                    ]
                    average = (
                        average / generic_verbalization_averages[verbalized_relation]
                    )

                confusion_matrix_values[true_relation][verbalized_relation] += values
                verbalized_relation_metric_averages[verbalized_relation] = average

            predicted_relation: Relation = min(
                verbalized_relation_metric_averages,
                key=verbalized_relation_metric_averages.get,
            )
            confusion_matrix_counts[true_relation][predicted_relation] += 1

    print(f"{confusion_matrix_counts=}")
    with open(confusion_matrix_counts_filepath, "w") as f:
        json.dump(confusion_matrix_counts, f)

    # print(f"{confusion_matrix_values=}")
    with open(confusion_matrix_values_filepath, "w") as f:
        json.dump(confusion_matrix_values, f)


if __name__ == "__main__":
    fire.Fire(main)
