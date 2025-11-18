from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # Set pad_token if not already set (some models use eos_token as pad_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here

        This would be a default implementation applies a basic chat template.
        Override this in subclasses for different behavior (e.g., SFT/RFT models should return raw questions).
        """
        # For BaseLLM, just return the question as-is
        # Subclasses (like CoTModel) will override this to use chat templates
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - apply format_prompt to the input prompt
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        # Use batched_generate for consistency - it generates the same way and has better loss
        # This ensures generate() and batched_generate() produce identical results
        results = self.batched_generate([prompt])
        return results[0] if results else ""

    @overload
    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: None = None,
        temperature: float = 0,
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(self, prompts: List[str]) -> List[str]:
        """
        Generate outputs for a batch of prompts.
        Ensures identical behavior to generate() and correct handling of left padding.
        """

        # 1. Format prompts
        formatted_prompts = [self.format_prompt(p) for p in prompts]

        # 2. Tokenize (left padding)
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.model.device)

        # Store the true prompt lengths (number of non-padding tokens)
        attention_mask = inputs["attention_mask"]
        prompt_lens = attention_mask.sum(dim=1).tolist()

        # 3. Call model.generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        # 4. Extract the newly generated tokens per sample
        results = []
        for i, prompt_len in enumerate(prompt_lens):
            # Slice exactly from the end of the prompt
            gen_tokens = outputs[i, prompt_len:]

            # Decode
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

            results.append(text)

        return results

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        generations = self.batched_generate(questions)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
