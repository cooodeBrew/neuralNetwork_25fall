from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Use arithmetic coding with probability distributions from autoregressive model.
        """
        device = x.device
        self.autoregressive.eval()
        
        # Tokenize the image
        with torch.no_grad():
            tokens = self.tokenizer.encode_index(x.unsqueeze(0) if x.dim() == 3 else x)  # (B, h, w)
        
        if tokens.dim() == 3:
            tokens = tokens[0] 
        
        h, w = tokens.shape
        tokens_flat = tokens.flatten().cpu().numpy().astype(np.int32)  # (seq_len,)
        seq_len = len(tokens_flat)
        
        result = bytearray()
        result.extend(self._int_to_bytes(h, 2))
        result.extend(self._int_to_bytes(w, 2))
        
        # Arithmetic coding using range coder
        # Use 31-bit precision to avoid overflow
        low = 0
        high = (1 << 31) - 1
        pending_bits = 0
        output_bits = []
        
        with torch.no_grad():
            current_tokens = torch.zeros(1, h, w, dtype=torch.long, device=device)
            
            for i in range(seq_len):
                pos_h, pos_w = i // w, i % w
                
                # Get probability distribution
                logits, _ = self.autoregressive(current_tokens)
                probs = torch.nn.functional.softmax(logits[0, pos_h, pos_w], dim=-1)
                probs_np = probs.cpu().numpy()
                
                # Normalize probabilities to avoid numerical issues
                probs_np = np.maximum(probs_np, 1e-10)
                probs_np = probs_np / probs_np.sum()
                
                # Compute cumulative probabilities
                cumprobs = np.cumsum(np.concatenate([[0.0], probs_np[:-1]]))
                cumprobs_end = np.cumsum(probs_np)
                
                token = tokens_flat[i]
                
                # Narrow range based on token probability
                range_size = high - low + 1
                token_low = low + int(range_size * cumprobs[token])
                token_high = low + int(range_size * cumprobs_end[token]) - 1
                
                # Ensure valid range
                if token_high < token_low:
                    token_high = token_low
                
                low = token_low
                high = token_high
                
                # Output bits when range allows
                while True:
                    if high < (1 << 30):  # Top quarter
                        # Output 0
                        output_bits.append(0)
                        for _ in range(pending_bits):
                            output_bits.append(1)
                        pending_bits = 0
                        low = (low << 1) & ((1 << 31) - 1)
                        high = ((high << 1) | 1) & ((1 << 31) - 1)
                    elif low >= (1 << 30):  # Bottom quarter
                        # Output 1
                        output_bits.append(1)
                        for _ in range(pending_bits):
                            output_bits.append(0)
                        pending_bits = 0
                        low = ((low - (1 << 30)) << 1) & ((1 << 31) - 1)
                        high = (((high - (1 << 30)) << 1) | 1) & ((1 << 31) - 1)
                    elif low >= (1 << 29) and high < (3 << 29):  # Middle half
                        # Underflow: increment pending
                        pending_bits += 1
                        low = ((low - (1 << 29)) << 1) & ((1 << 31) - 1)
                        high = (((high - (1 << 29)) << 1) | 1) & ((1 << 31) - 1)
                    else:
                        break
                
                # Update current sequence for next prediction
                current_tokens[0, pos_h, pos_w] = token
        
        # Flush remaining bits
        pending_bits += 1
        if low < (1 << 29):
            output_bits.append(0)
            for _ in range(pending_bits):
                output_bits.append(1)
        else:
            output_bits.append(1)
            for _ in range(pending_bits):
                output_bits.append(0)
        
        # Convert bits to bytes
        for i in range(0, len(output_bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(output_bits):
                    byte |= (output_bits[i + j] << (7 - j))
            result.append(byte)
        
        return bytes(result)
    
    def _int_to_bytes(self, value: int, num_bytes: int) -> bytes:
        """Convert integer to bytes."""
        return value.to_bytes(num_bytes, byteorder='big')
    
    def _bytes_to_int(self, data: bytes) -> int:
        """Convert bytes to integer."""
        return int.from_bytes(data, byteorder='big')

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        You may assume the output image is 150 x 100 pixels.
        """
        device = next(self.autoregressive.parameters()).device
        self.autoregressive.eval()
        
        # Parse header
        h = self._bytes_to_int(x[0:2])
        w = self._bytes_to_int(x[2:4])
        seq_len = h * w
        
        # Convert bytes to bit stream
        bit_stream = []
        for byte in x[4:]:
            for bit in range(8):
                bit_stream.append((byte >> (7 - bit)) & 1)
        
        # Initialize decoder
        # by cursor
        low = 0
        high = (1 << 31) - 1
        value = 0
        for i in range(min(31, len(bit_stream))):
            value = (value << 1) | bit_stream[i]
        bit_index = 31
        
        tokens_flat = []
        
        with torch.no_grad():
            current_tokens = torch.zeros(1, h, w, dtype=torch.long, device=device)
            
            for i in range(seq_len):
                pos_h, pos_w = i // w, i % w
                
                # Get probability distribution
                logits, _ = self.autoregressive(current_tokens)
                probs = torch.nn.functional.softmax(logits[0, pos_h, pos_w], dim=-1)
                probs_np = probs.cpu().numpy()
                
                # Normalize probabilities
                probs_np = np.maximum(probs_np, 1e-10)
                probs_np = probs_np / probs_np.sum()
                
                # Compute cumulative probabilities
                cumprobs = np.cumsum(np.concatenate([[0.0], probs_np[:-1]]))
                cumprobs_end = np.cumsum(probs_np)
                
                # Find token that matches current value
                range_size = high - low + 1
                scaled_value = (value - low) / range_size
                
                token = 0
                for t in range(len(probs_np)):
                    if cumprobs[t] <= scaled_value < cumprobs_end[t]:
                        token = t
                        break
                
                tokens_flat.append(token)
                
                # Narrow range
                token_low = low + int(range_size * cumprobs[token])
                token_high = low + int(range_size * cumprobs_end[token]) - 1
                
                if token_high < token_low:
                    token_high = token_low
                
                low = token_low
                high = token_high
                
                # Expand range and read more bits
                while True:
                    if high < (1 << 30):
                        low = (low << 1) & ((1 << 31) - 1)
                        high = ((high << 1) | 1) & ((1 << 31) - 1)
                        if bit_index < len(bit_stream):
                            value = ((value << 1) | bit_stream[bit_index]) & ((1 << 31) - 1)
                            bit_index += 1
                    elif low >= (1 << 30):
                        low = ((low - (1 << 30)) << 1) & ((1 << 31) - 1)
                        high = (((high - (1 << 30)) << 1) | 1) & ((1 << 31) - 1)
                        if bit_index < len(bit_stream):
                            value = (((value - (1 << 30)) << 1) | bit_stream[bit_index]) & ((1 << 31) - 1)
                            value += (1 << 30)
                            bit_index += 1
                    elif low >= (1 << 29) and high < (3 << 29):
                        low = ((low - (1 << 29)) << 1) & ((1 << 31) - 1)
                        high = (((high - (1 << 29)) << 1) | 1) & ((1 << 31) - 1)
                        if bit_index < len(bit_stream):
                            value = (((value - (1 << 29)) << 1) | bit_stream[bit_index]) & ((1 << 31) - 1)
                            value += (1 << 29)
                            bit_index += 1
                    else:
                        break
                
                # Update current sequence for next prediction
                # cursor part ends
                current_tokens[0, pos_h, pos_w] = token
        
        # Convert to tensor and decode
        tokens = torch.tensor(tokens_flat[:seq_len], dtype=torch.long, device=device).view(h, w)
        tokens = tokens.unsqueeze(0)
        
        # Decode tokens to image
        with torch.no_grad():
            image = self.tokenizer.decode_index(tokens)
        
        # Remove batch dimension
        return image[0]  


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
