docker run --name tcs --rm -it -v "$PWD:/app" --env-file .env --runtime nvidia tcs python -u measure_common_sense.py --lm_mode causal --model_id meta-llama/Meta-Llama-3.1-8B #--quantization
