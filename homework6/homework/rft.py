from .base_llm import BaseLLM
from .sft import test_model


class RFTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        RFT models are trained on raw questions without chat templates.
        Return the question as-is.
        """
        return question


def load() -> RFTModel:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = RFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str = "./homework/rft_model",
    **kwargs,
):
    import json
    from pathlib import Path
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    from .sft import TokenizedDataset, tokenize
    
    # Load RFT data from JSON file
    # Check both possible locations (data/rft.json and data/data/rft.json)
    data_path = Path(__file__).parent.parent / "data" / "rft.json"
    if not data_path.exists():
        # Try the nested location (in case user passed "data/rft.json" to datagen)
        data_path_nested = Path(__file__).parent.parent / "data" / "data" / "rft.json"
        if data_path_nested.exists():
            data_path = data_path_nested
            print(f"Found RFT data at nested location: {data_path}")
        else:
            raise FileNotFoundError(
                f"RFT data file not found at {data_path} or {data_path_nested}. "
                f"Please run: python -m homework.datagen rft.json"
            )
    
    with open(data_path, 'r') as f:
        rft_data = json.load(f)
    
    # Create a Dataset-like object for RFT data
    class RFTDataset:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            # RFT data format: [question, answer, reasoning]
            return self.data[idx]
    
    train_data = RFTDataset(rft_data)
    
    # Initialize model
    llm = RFTModel()
    
    # Set up LoRA configuration - slightly larger for RFT
    # Use r=32 to allow for better reasoning capabilities (still keep under 50MB total)
    lora_config = LoraConfig(
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
        r=32,  # Larger rank for better reasoning
        lora_alpha=128,  # About 4x the rank
        lora_dropout=0.1,
    )
    
    # Convert model to LoRA
    llm.model = get_peft_model(llm.model, lora_config)
    
    # Enable input require grads to avoid gradient checkpointing bug
    if llm.device != "cpu":
        llm.model.enable_input_require_grads()
    
    # Format function for RFT data
    # RFT data format: [question, answer, reasoning]
    # We use the reasoning (which includes the answer) as the training target
    def format_rft_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
        # The reasoning already contains the answer in <answer> tags
        # So we can use it directly
        return {"question": question, "answer": reasoning}
    
    # Create tokenized dataset
    tokenized_dataset = TokenizedDataset(
        tokenizer=llm.tokenizer,
        data=train_data,
        format_fn=format_rft_example
    )
    
    # Set up training arguments
    # Disable gradient checkpointing on CPU (it's slower and not needed)
    # On GPU, keep it enabled to save memory
    if 'gradient_checkpointing' not in kwargs:
        kwargs['gradient_checkpointing'] = llm.device != "cpu"
    gradient_checkpointing = kwargs.pop('gradient_checkpointing', llm.device != "cpu")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        gradient_checkpointing=gradient_checkpointing,  # Save GPU memory, but disable on CPU
        learning_rate=5e-4,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        save_strategy="no",  # Don't save checkpoints during training to save space
        logging_steps=10,
        **kwargs
    )
    
    # Create trainer
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train
    trainer.train()
    
    # Save only the LoRA adapter (not the full model) to keep size reasonable
    output_path = Path(__file__).parent / "rft_model"
    llm.model.save_pretrained(str(output_path))
    
    # Clean up training checkpoints to save space
    import shutil
    checkpoint_dirs = [d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
    for checkpoint_dir in checkpoint_dirs:
        shutil.rmtree(checkpoint_dir)
        print(f"Deleted checkpoint: {checkpoint_dir}")
    
    # Test the model
    test_model(str(output_path))


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
