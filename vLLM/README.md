







vllm 






# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --model BAAI/bge-large-en-v1.5 \
  --backend openai-embeddings \
  --endpoint /v1/embeddings \
  --dataset-name random \
  --random-input-len 256 \
  --max-concurrency 32 \
  --num-prompts 100 \
  --request-rate inf