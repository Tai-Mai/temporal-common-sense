# Temporal Common Sense Testing of Pre-Trained Language Models
This repository contains the code to test pre-trained language models for their common sense on Allen's interval algebra.
The associated research report can be found in `./report`

## Options
* Set the `--model_id` to the huggingface model id of the desired model. Tested examples:
  * `meta-llama/Meta-Llama-3.1-8B`
  * `roberta-base`
  * `gpt2`
* Make sure to set `--lm_mode` to the correct kind of language modeling according to the model specified with `--model_id`. Can be `causal` or `masked`.
* Add the `--quantization` flag to enable 4-bit quantization

## Usage
### Docker
This repository does not use `docker compose` since it it's not available on Google Cloud, which is what I was using to run this. Therefore, we run it using a `docker run` command that's saved in `./start_script.sh`
1. Build the image
  ```sh
  $ docker build -t tcs .
  ```
2. Make the run script executable
  ```sh
  $ chmod +x ./start_script.sh
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
  ```sh
  $ pip install -r requirements.txt
  ```
2. Optional: Create an `.env` file and add the following line with your huggingface token. This is needed for gated models such as the Llama model family
  ```
  HF_TOKEN=<your_token>
  ```
3. Run the script
  ```sh
  $ python measure_common_sense.py --lm_mode causal --model_id meta-llama/Meta-Llama-3.1-8B
  ```
  * Or change the parameters as described above
