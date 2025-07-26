import torch
from transformers import pipeline

# Create the fill-mask pipeline
fill_mask = pipeline(
    task="fill-mask",
    model="google-bert/bert-base-uncased",
    torch_dtype=torch.float16,
    device=0  # assumes you have a CUDA-capable GPU
)

# Test the model
result = fill_mask(" I love [MASK].")

# Print the top predictions
for i, prediction in enumerate(result):
    print(f"{i+1}: {prediction['sequence']} (score: {prediction['score']:.4f})")
