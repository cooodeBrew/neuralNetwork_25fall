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
    # Round the answer to 3 decimal places to make it easier for the LLM
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
    
    train_dataset = Dataset("train")
    
    model = SFTModel()
    
    # Using r=16 to keep model size below 20MB (can adjust if needed)
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=64,  # 4 * r
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Convert model to LoRA
    model.model = get_peft_model(model.model, lora_config)
    
    # Enable input require grads for gradient checkpointing (GPU fix)
    if model.device == "cuda":
        model.model.enable_input_require_grads()
    
    tokenized_dataset = TokenizedDataset(
        tokenizer=model.tokenizer,
        data=train_dataset,
        format_fn=format_example
    )
    
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
    
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    
    test_model(output_dir)


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
