from .base_llm import BaseLLM
from .data import Dataset, benchmark


class SFTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        SFT models are trained on raw questions without chat templates.
        Return the question as-is.
        """
        return question


def load() -> SFTModel:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = SFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # Round the answer to make it easier for the LLM
    rounded_answer = round(answer, 3)
    formatted_answer = f"<answer>{rounded_answer}</answer>"
    return {"question": prompt, "answer": formatted_answer}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str = "./homework/sft_model",
    **kwargs,
):
    from pathlib import Path
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    
    # Load data
    train_data = Dataset("train")
    
    # Initialize model
    llm = SFTModel()
    
    # Print device information
    print(f"Training on device: {llm.device}")
    if llm.device == "cpu":
        print("WARNING: Training on CPU is very slow! Consider using GPU if available.")
        print("Expected time on CPU: 5-15 hours. On GPU: 30-90 minutes.")
    
    # Set up LoRA configuration
    # Use r=16 to keep model size below 20MB (r=16, alpha=64 gives reasonable size)
    lora_config = LoraConfig(
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
        r=16,  # Rank - adjust to keep model size < 20MB
        lora_alpha=64,  # About 4x the rank
        lora_dropout=0.1,
    )
    
    # Convert model to LoRA
    llm.model = get_peft_model(llm.model, lora_config)
    
    # Enable input require grads BEFORE setting training mode
    # This is critical for gradient checkpointing to work properly
    llm.model.enable_input_require_grads()
    
    # Set model to training mode
    llm.model.train()
    
    # Print trainable parameters for debugging
    trainable_params = sum(p.numel() for p in llm.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in llm.model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Create tokenized dataset
    tokenized_dataset = TokenizedDataset(
        tokenizer=llm.tokenizer,
        data=train_data,
        format_fn=format_example
    )
    
    # Set up training arguments
    # Disable gradient checkpointing on CPU (it's slower and not needed)
    # On GPU, keep it enabled to save memory
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
    
    # Save only the LoRA adapter (not the full model) to keep size under 20MB
    output_path = Path(__file__).parent / "sft_model"
    llm.model.save_pretrained(str(output_path))
    
    # Clean up training checkpoints to save space
    import shutil
    checkpoint_dirs = [d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
    for checkpoint_dir in checkpoint_dirs:
        shutil.rmtree(checkpoint_dir)
        print(f"Deleted checkpoint: {checkpoint_dir}")
    
    # Test the model
    test_model(str(output_path))


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = SFTModel()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
