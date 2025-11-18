"""
Test script to check the actual loss values for generate() and batched_generate()
This mimics what the grader does to help debug the issue.
"""
import torch
from homework.base_llm import BaseLLM
from homework.data import Dataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def compute_loss(model, full_texts):
    """
    Compute the loss of the model on the full texts (same as grader).
    """
    with torch.no_grad():
        tokens = model.tokenizer(full_texts, return_tensors="pt", padding=True)
        answer_output = model.model(
            input_ids=tokens["input_ids"].to(device),
            attention_mask=tokens["attention_mask"].to(device),
        )
        logits = answer_output.logits
        logits = logits[..., :-1, :].contiguous()
        labels = tokens["input_ids"][..., 1:].contiguous().to(device)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )

        loss = loss * tokens["attention_mask"][..., 1:].contiguous().to(device)
        loss = loss.sum() / tokens["attention_mask"][..., 1:].sum()
        return loss.cpu().item()

def test_generate_loss():
    """Test generate() method loss"""
    print("=" * 60)
    print("Testing generate() method (non-batched)")
    print("=" * 60)
    
    llm = BaseLLM()
    llm.model.eval()
    
    dataset = Dataset("valid")
    test_size = 32  # Same as grader
    questions = [dataset[i][0] for i in range(min(test_size, len(dataset)))]
    
    print(f"Testing on {len(questions)} questions...")
    
    # Generate answers using generate()
    answers = []
    for i, question in enumerate(questions):
        answer = llm.generate(question)
        answers.append(answer)
        if i < 3:  # Print first 3 examples
            print(f"\nExample {i+1}:")
            print(f"  Question: {question}")
            print(f"  Generated: {answer[:100]}...")  # First 100 chars
    
    # Compute loss
    full_texts = [questions[i] + answers[i] for i in range(len(questions))]
    loss = compute_loss(llm, full_texts)
    
    print(f"\n{'='*60}")
    print(f"Loss: {loss:.4f}")
    print(f"Expected range: 6.2 - 8.0")
    print(f"Score calculation:")
    if loss <= 6.2:
        score = 1.0
        print(f"  Loss ≤ 6.2 → Full points (1.0)")
    elif loss >= 8.0:
        score = 0.0
        print(f"  Loss ≥ 8.0 → Zero points (0.0)")
    else:
        score = 1.0 - (loss - 6.2) / (8.0 - 6.2)
        print(f"  Loss between 6.2-8.0 → Linear interpolation")
    print(f"  Final score: {score:.4f} (out of 1.0)")
    print(f"{'='*60}\n")
    
    return loss

def test_batched_generate_loss():
    """Test batched_generate() method loss"""
    print("=" * 60)
    print("Testing batched_generate() method")
    print("=" * 60)
    
    llm = BaseLLM()
    llm.model.eval()
    
    dataset = Dataset("valid")
    test_size = 32  # Same as grader
    questions = [dataset[i][0] for i in range(min(test_size, len(dataset)))]
    
    print(f"Testing on {len(questions)} questions...")
    
    # Generate answers using batched_generate()
    answers = llm.batched_generate(questions)
    
    # Print first 3 examples
    for i in range(min(3, len(questions))):
        print(f"\nExample {i+1}:")
        print(f"  Question: {questions[i]}")
        print(f"  Generated: {answers[i][:100]}...")  # First 100 chars
    
    # Compute loss
    full_texts = [questions[i] + answers[i] for i in range(len(questions))]
    loss = compute_loss(llm, full_texts)
    
    print(f"\n{'='*60}")
    print(f"Loss: {loss:.4f}")
    print(f"Expected range: 6.2 - 8.0")
    print(f"Score calculation:")
    if loss <= 6.2:
        score = 1.0
        print(f"  Loss ≤ 6.2 → Full points (1.0)")
    elif loss >= 8.0:
        score = 0.0
        print(f"  Loss ≥ 8.0 → Zero points (0.0)")
    else:
        score = 1.0 - (loss - 6.2) / (8.0 - 6.2)
        print(f"  Loss between 6.2-8.0 → Linear interpolation")
    print(f"  Final score: {score:.4f} (out of 1.0)")
    print(f"{'='*60}\n")
    
    return loss

if __name__ == "__main__":
    print("\nTesting BaseLLM Loss Values\n")
    
    # Test generate()
    loss1 = test_generate_loss()
    
    # Test batched_generate()
    loss2 = test_batched_generate_loss()
    
    print("\nSummary:")
    print(f"  generate() loss:        {loss1:.4f}")
    print(f"  batched_generate() loss: {loss2:.4f}")
    print(f"  Expected range:         6.2 - 8.0")
    print(f"\nIf loss > 8.0, you get 0 points")
    print(f"If loss < 6.2, you get full points")
    print(f"If 6.2 ≤ loss ≤ 8.0, score is interpolated")

