def generate_dataset(output_json: str = "rft.json", oversample: int = 10, temperature: float = 0.6):
    import json
    from pathlib import Path
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid
    
    # Use the larger model for better rollouts
    checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    
    # Create CoT model with the larger checkpoint
    class CoTModelLarge(CoTModel):
        def __init__(self):
            from .base_llm import device
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            # Set pad_token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
            self.device = device
    
    model = CoTModelLarge()
    model.model.eval()
    
    # Load training data
    train_data = Dataset("train")
    
    # Generate dataset
    rft_data = []
    
    for question, correct_answer in train_data:
        # Generate multiple completions
        completions = model.batched_generate(
            [question],
            num_return_sequences=oversample,
            temperature=temperature
        )[0]  # Get the list of completions for this question
        
        # Find the first correct answer
        found_correct = False
        for completion in completions:
            # Parse the answer from the completion
            parsed_answer = model.parse_answer(completion)
            
            # Check if the answer is valid
            if is_answer_valid(parsed_answer, correct_answer):
                # Found a correct answer, add to dataset
                rft_data.append([question, correct_answer, completion])
                found_correct = True
                break
        
        # If no correct answer found, skip this data point
        if not found_correct:
            continue
    
    # Save to JSON file
    output_path = Path(__file__).parent.parent / "data" / output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(rft_data, f, indent=2)
    
    print(f"Generated {len(rft_data)} examples and saved to {output_path}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
