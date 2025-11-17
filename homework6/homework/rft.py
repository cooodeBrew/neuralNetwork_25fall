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
    
    from .sft import tokenize
    
    # Load RFT dataset
    rft_data_path = Path(__file__).parent.parent / "data" / "rft.json"
    if not rft_data_path.exists():
        raise FileNotFoundError(
            f"RFT dataset not found at {rft_data_path}. "
            "Please run 'python -m homework.datagen data/rft.json' first."
        )
    
    with rft_data_path.open() as f:
        rft_data = json.load(f)
    
    # Initialize model
    model = RFTModel()
    
    # Configure LoRA (can use slightly larger rank for RFT)
    # Using r=32 to allow for better performance while staying under 50MB
    lora_config = LoraConfig(
        r=32,  # rank (larger than SFT for better performance)
        lora_alpha=128,  # 4 * r
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Convert model to LoRA
    model.model = get_peft_model(model.model, lora_config)
    
    # Enable input require grads for gradient checkpointing (GPU fix)
    if model.device.type == "cuda":
        model.model.enable_input_require_grads()
    
    # Format function for RFT data: [question, answer, reasoning]
    def format_rft_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
        """
        Format RFT example: question + reasoning (which includes the answer)
        The reasoning already contains <answer>...</answer>, so we use it directly
        """
        return {"question": question, "answer": reasoning}
    
    # Create a simple dataset wrapper for RFT data
    # by cursor
    class RFTDataset:
        def __init__(self, data, format_fn):
            self.data = data
            self.format_fn = format_fn
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            # RFT data format: [question, answer, reasoning]
            question, answer, reasoning = self.data[idx]
            formatted_data = self.format_fn(question, answer, reasoning)
            return tokenize(model.tokenizer, **formatted_data)
    
    # Create tokenized dataset
    tokenized_dataset = RFTDataset(
        data=rft_data,
        format_fn=format_rft_example
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        learning_rate=5e-4,
        gradient_checkpointing=True,
        save_strategy="epoch",
        save_total_limit=1,  # Only keep the last checkpoint to save space
    )
    
    # Create trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    
    # Save the final model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    
    # Test the model
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
