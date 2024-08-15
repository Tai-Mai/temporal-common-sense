# Temporal Common Sense Testing of Pre-Trained Language Models
This repository contains the code and data to test pre-trained language models for their common sense on Allen's interval algebra.
The associated research report can be found in [`./report/report.pdf`](https://github.com/Tai-Mai/temporal-common-sense/blob/main/report/report.pdf). The dataset can be found in [`./data/claude_examples.json`](https://github.com/Tai-Mai/temporal-common-sense/blob/main/data/claude_examples.json). 

## Options
* `--model_id`: Set to the huggingface model id of the desired model. Tested examples:
  * `meta-llama/Meta-Llama-3.1-8B`
  * `roberta-base`
  * `gpt2`
* `--lm_mode`: Make sure to set to the correct kind of language modeling according to the model specified with `--model_id`. Can be one of the following:
  * `causal` 
  * `masked`
* `--quantization`: Add this flag to enable 4-bit quantization
* `--normalize`: Add this flag to normalize the metric scores by subtracting the generic scores of the verbalizations. This aims to cancel out the effect that verbalizations that have an inherently low metric score will drag down the verbalized scores as well

## Usage
### Docker
This repository does not use `docker compose` since it it's not available on Google Cloud, which is what I was using to run this. Therefore, we run it using a `docker run` command that's saved in `./start_script.sh`
1. Build the image
  ```bash
  $ docker build -t tcs .
  ```
2. Make the run script executable
  ```bash
  $ chmod u+x ./start_script.sh
  ```
3. Edit `./start_script.sh`
  * Change the `--model_id` to the desired model to test
  * Make sure to set `--lm_mode` to the correct kind of language modeling. Can be `causal` or `masked`.
  * Add the `--quantization` flag to enable 4-bit quantization
4. Optional: Create an `.env` file and add the following line with your huggingface token. This is needed for gated models such as the Llama model family
  ```
  HF_TOKEN=<your_token>
  ```
5. Docker run
  ```sh
  $ ./start_script.sh
  ```

### Manually
1. Install dependencies
  ```bash
  $ pip install -r requirements.txt
  ```
2. Optional: Create an `.env` file and add the following line with your huggingface token. This is needed for gated models such as the Llama model family
  ```
  HF_TOKEN=<your_token>
  ```
3. Run the script
  ```bash
  $ python measure_common_sense.py --lm_mode causal --model_id meta-llama/Meta-Llama-3.1-8B
  ```
  * Or change the parameters as described above

### Evaluation
* Plot the confusion matrix heatmaps
  ```bash
  $ python plot_confusion_matrices.py path/to/<confusion_matrix_name>.json
  ```
  * PDF plot will be saved to `confusion_matrices/plots/<confusion_matrix_name>.pdf`
* Calculate the correlation coefficients between graph hops and perplexity
  ```bash
  $ python correlation.py path/to/<confusion_matrix_name>.json
  ```
  * `--deltas`: Use perplexity deltas instead of absolute perplexity values
