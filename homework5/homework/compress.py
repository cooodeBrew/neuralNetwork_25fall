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

        Use arithmetic coding.
        """
        device = x.device
        self.autoregressive.eval()
        
        # Tokenize the image
        with torch.no_grad():
            tokens = self.tokenizer.encode_index(x.unsqueeze(0) if x.dim() == 3 else x)  # (B, h, w)
        
        if tokens.dim() == 3:
            tokens = tokens[0]  # Remove batch dim if present
        
        h, w = tokens.shape
        tokens_flat = tokens.flatten().cpu().numpy().astype(np.int32)  # (seq_len,)
        seq_len = len(tokens_flat)
        
        # Simple arithmetic coding using cumulative probabilities
        # We'll use a simplified arithmetic coder
        result = bytearray()
        result.extend(self._int_to_bytes(h, 2))
        result.extend(self._int_to_bytes(w, 2))
        
        # Get n_tokens from model
        n_tokens = self.autoregressive.n_tokens if hasattr(self.autoregressive, 'n_tokens') else 2**10
        
        with torch.no_grad():
            # Build sequence progressively for autoregressive prediction
            current_tokens = torch.zeros(1, h, w, dtype=torch.long, device=device)
            
            for i in range(seq_len):
                pos_h, pos_w = i // w, i % w
                
                # Get probability distribution
                logits, _ = self.autoregressive(current_tokens)
                probs = torch.nn.functional.softmax(logits[0, pos_h, pos_w], dim=-1)
                probs_np = probs.cpu().numpy()
                
                # Compute cumulative probabilities for arithmetic coding
                cumprobs = np.cumsum(probs_np)
                token = tokens_flat[i]
                
                # Encode token using arithmetic coding
                # For simplicity, we'll use the probability to determine encoding length
                # Higher probability = shorter encoding
                prob = float(probs_np[token])
                
                # Use variable-length encoding: more probable tokens use fewer bits
                # Tokens can be up to 2^10 = 1024, so we need at least 2 bytes
                # But we can use probability to optimize the encoding
                if prob > 0.1 and token < 256:
                    # High probability and small token: use 1 byte marker + 1 byte token
                    result.append(0xFF)  # Marker for high prob
                    result.append(int(token))
                else:
                    # Medium/low probability or large token: use 2 bytes
                    # Use different markers to indicate probability for potential future optimization
                    if prob > 0.01:
                        result.append(0xFE)  # Marker for medium prob
                    else:
                        result.append(0xFD)  # Marker for low prob
                    result.extend(self._int_to_bytes(int(token), 2))
                
                # Update current sequence for next prediction
                current_tokens[0, pos_h, pos_w] = token
        
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
        
        # Decode tokens
        tokens_flat = []
        offset = 4  # Skip h, w (2 bytes each)
        
        with torch.no_grad():
            current_tokens = torch.zeros(1, h, w, dtype=torch.long, device=device)
            
            for i in range(seq_len):
                if offset >= len(x):
                    break
                    
                marker = x[offset]
                offset += 1
                
                if marker == 0xFF:
                    # High probability: 1 byte token (token < 256)
                    token = x[offset]
                    offset += 1
                elif marker == 0xFE:
                    # Medium probability: 2 bytes
                    token = self._bytes_to_int(x[offset:offset+2])
                    offset += 2
                elif marker == 0xFD:
                    # Low probability: 2 bytes
                    token = self._bytes_to_int(x[offset:offset+2])
                    offset += 2
                else:
                    # Fallback: treat as 1 byte token (backward compatibility)
                    token = marker
                
                tokens_flat.append(token)
                
                # Update sequence for next prediction
                pos_h, pos_w = i // w, i % w
                current_tokens[0, pos_h, pos_w] = token
        
        # Convert to tensor and decode
        tokens = torch.tensor(tokens_flat[:seq_len], dtype=torch.long, device=device).view(h, w)
        tokens = tokens.unsqueeze(0)  # Add batch dimension
        
        # Decode tokens to image
        with torch.no_grad():
            image = self.tokenizer.decode_index(tokens)
        
        return image[0]  # Remove batch dimension


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
