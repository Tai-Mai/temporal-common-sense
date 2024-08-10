#docker run -d --name tcs --rm -it -v "$PWD:/app" --env-file .env --runtime nvidia tcs python -u measure_common_sense.py --lm_mode causal --model_id meta-llama/Meta-Llama-3.1-8B #--quantization
#docker run -d --name tcs --rm -it -v "$PWD:/app" --env-file .env --runtime nvidia tcs python -u measure_common_sense.py --lm_mode masked --model_id roberta-base #--quantization
docker run -d --name tcs --rm -it -v "$PWD:/app" --env-file .env --runtime nvidia tcs python -u measure_common_sense.py --lm_mode causal --model_id gpt2 #--quantization
OUTPUT_FILE=output.txt
if [ -f "$OUTPUT_FILE" ]; then
    # Rename the file
    mv "$OUTPUT_FILE" "old_$OUTPUT_FILE"
    echo "$OUTPUT_FILE already exists. Renamed to old_$OUTPUT_FILE."
fi
echo "Output is redirected to $OUTPUT_FILE"
docker logs -f tcs > $OUTPUT_FILE &
