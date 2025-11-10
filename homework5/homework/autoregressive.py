import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        
        # Token embedding
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        
        # Transformer layers with causal masking
        # Use TransformerEncoderLayer with causal mask
        self.transformer_layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=d_latent,
                nhead=8,
                dim_feedforward=4 * d_latent,
                dropout=0.1,
                batch_first=True
            ) for _ in range(4)
        ])
        
        # Output projection to vocabulary
        self.output_proj = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        x: (B, h, w) integer tokens
        Returns: (B, h, w, n_tokens) logits and additional losses dict
        """
        B, h, w = x.shape
        seq_len = h * w
        
        # Flatten to sequence: (B, h, w) -> (B, seq_len)
        x_flat = x.view(B, seq_len)
        
        # Embed tokens: (B, seq_len) -> (B, seq_len, d_latent)
        x_emb = self.embedding(x_flat)
        
        # Shift input by 1 position for autoregressive prediction
        # Pad with zeros at the beginning
        x_shifted = torch.cat([
            torch.zeros(B, 1, self.d_latent, device=x.device, dtype=x_emb.dtype),
            x_emb[:, :-1]
        ], dim=1)
        
        # Create causal mask: prevent attending to future tokens
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        # Apply transformer layers
        out = x_shifted
        for layer in self.transformer_layers:
            out = layer(out, src_mask=mask)
        
        # Project to vocabulary: (B, seq_len, d_latent) -> (B, seq_len, n_tokens)
        logits = self.output_proj(out)
        
        # Reshape back to spatial: (B, seq_len, n_tokens) -> (B, h, w, n_tokens)
        logits = logits.view(B, h, w, self.n_tokens)
        
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Autoregressively generate tokens.
        Returns: (B, h, w) integer tokens
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        seq_len = h * w
        
        # Initialize with zeros (or random tokens)
        generated = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for pos in range(seq_len):
                # Get logits for current position
                # We need to process the sequence up to current position
                current_seq = generated[:, :pos+1]  # (B, pos+1)
                
                # Embed
                x_emb = self.embedding(current_seq)  # (B, pos+1, d_latent)
                
                # Shift (pad at beginning)
                if pos == 0:
                    x_shifted = torch.zeros(B, 1, self.d_latent, device=device, dtype=x_emb.dtype)
                else:
                    x_shifted = torch.cat([
                        torch.zeros(B, 1, self.d_latent, device=device, dtype=x_emb.dtype),
                        x_emb[:, :-1]
                    ], dim=1)
                
                # Create mask for current sequence length
                mask = torch.nn.Transformer.generate_square_subsequent_mask(pos+1, device=device)
                
                # Apply transformer
                out = x_shifted
                for layer in self.transformer_layers:
                    out = layer(out, src_mask=mask)
                
                # Get logits for next token (last position)
                logits = self.output_proj(out[:, -1])  # (B, n_tokens)
                
                # Sample from distribution
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
                
                # Store generated token
                generated[:, pos] = next_token
        
        # Reshape to spatial: (B, seq_len) -> (B, h, w)
        return generated.view(B, h, w)
