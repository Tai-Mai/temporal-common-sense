import fire
import json
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    PreTrainedModel,
)
from tqdm import tqdm

from utils.metrics import Metric, Perplexity, PseudoPerplexity


type Relation = str
type Verbalization = str
type Event = str
type Count = int


def main(lm_mode: str = "causal", model_id: str = "", quantization: bool = False):
    match lm_mode:
        case "causal":
            model_id = model_id or "gpt2"
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_id)
            metric: Metric = Perplexity(model=model)
        case "masked":
            model_id = model_id or "roberta-base"
            model: PreTrainedModel = AutoModelForMaskedLM.from_pretrained(model_id)
            metric: Metric = PseudoPerplexity(model=model)
        case _:
            raise ValueError("Invalid language modeling mode")

    if quantization:
        model = prepare_model_for_kbit_training(
            model, gradient_checkpointing_kwargs={"use_reentrant": True}
        )
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            use_rslora=True,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            inference_mode=True,
        )

        model: PeftModel = get_peft_model(model, peft_config=peft_config)

    data_filepath = "data/claude_examples.json"
    output_filepath = f"confusion_matrices_{model_id}.json"

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
            predicted_relation: Relation = min(
                relation_metric_values, key=relation_metric_values.get
            )
            confusions[predicted_relation] += 1
        confusion_matrices[true_relation] = confusions

    print(confusion_matrices)
    with open(output_filepath, "w") as f:
        json.dump(confusion_matrices, f)


if __name__ == "__main__":
    fire.Fire(main)
