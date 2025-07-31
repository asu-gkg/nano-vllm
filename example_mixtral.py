import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # Update path to your Mixtral model
    path = os.path.expanduser("~/models/mixtral-8x7b")  # Change to your model path
    
    # Note: Mixtral models can be large, ensure you have enough GPU memory
    # For testing, you might want to use a smaller variant like Mixtral-8x7B-Instruct-v0.1
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
    prompts = [
        "What is machine learning?",
        "Explain the concept of mixture of experts in neural networks.",
    ]
    
    # Apply chat template if the model uses one
    if hasattr(tokenizer, "apply_chat_template"):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n" + "="*50)
        print(f"Prompt: {prompt}")
        print(f"Response: {output['text']}")


if __name__ == "__main__":
    main()