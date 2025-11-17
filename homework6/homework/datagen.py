def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    import json
    from pathlib import Path
    
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid
    
    train_dataset = Dataset("train")
    
    model = CoTModel()
    
    generated_data = []
    
    print(f"Generating RFT dataset with oversample={oversample}, temperature={temperature}")
    
    for idx, (question, correct_answer) in enumerate(train_dataset):
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(train_dataset)} examples...")
        
        generations = model.batched_generate(
            [question],
            num_return_sequences=oversample,
            temperature=temperature
        )
        
        completions = generations[0]
        
        # Find the first completion with a correct answer
        found_correct = False
        for completion in completions:
            parsed_answer = model.parse_answer(completion)
            
            if parsed_answer == parsed_answer and is_answer_valid(parsed_answer, correct_answer):
                generated_data.append([question, correct_answer, completion])
                found_correct = True
                break
        
        if not found_correct:
            # Skip this data point if no correct answer found
            continue
    
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w") as f:
        json.dump(generated_data, f, indent=2)
    
    print(f"Generated {len(generated_data)} examples out of {len(train_dataset)}")
    print(f"Saved to {output_json}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
