"""
Usage:
    python3 -m homework.train_planner --model mlp_planner
    python3 -m homework.train_planner --model transformer_planner
    python3 -m homework.train_planner --model vit_planner
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric
from homework.models import MODEL_FACTORY, save_model


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    metric = PlannerMetric()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        # Move data to device
        track_left = batch["track_left"].to(device)
        track_right = batch["track_right"].to(device)
        waypoints = batch["waypoints"].to(device)
        waypoints_mask = batch["waypoints_mask"].to(device)

        # Forward pass
        optimizer.zero_grad()
        
        # Get model predictions
        if "ViTPlanner" in model.__class__.__name__:
            image = batch["image"].to(device)
            pred = model(image=image)
        else:
            pred = model(track_left=track_left, track_right=track_right)

        # Compute loss
        loss = criterion(pred, waypoints)
        # Apply mask if needed
        mask = waypoints_mask.unsqueeze(-1)
        loss = (loss * mask).sum() / mask.sum()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        with torch.no_grad():
            metric.add(pred.detach().cpu(), waypoints.cpu(), waypoints_mask.cpu())
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches, metric.compute()


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    metric = PlannerMetric()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move data to device
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            # Get model predictions
            if "ViTPlanner" in model.__class__.__name__:
                image = batch["image"].to(device)
                pred = model(image=image)
            else:
                pred = model(track_left=track_left, track_right=track_right)

            # Compute loss
            loss = criterion(pred, waypoints)
            mask = waypoints_mask.unsqueeze(-1)
            loss = (loss * mask).sum() / mask.sum()

            # Update metrics
            metric.add(pred.cpu(), waypoints.cpu(), waypoints_mask.cpu())
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches, metric.compute()


def train(
    model_name: str,
    transform_pipeline: str = None,
    num_workers: int = 4,
    lr: float = 1e-3,
    batch_size: int = 32,
    num_epoch: int = 50,
    device: str = None,
    seed: int = 42,
):
    """
    Train a planner model. This function can be called from Colab or other environments.
    
    Args:
        model_name: Name of the model to train (e.g., "mlp_planner", "transformer_planner", "vit_planner")
        transform_pipeline: Which transform pipeline to use ("default" or "state_only")
        num_workers: Number of data loading workers
        lr: Learning rate
        batch_size: Batch size
        num_epoch: Number of epochs
        device: Device to use (None for auto-detect)
        seed: Random seed
    """
    # Map model_name to expected format if needed
    model_name_map = {
        "linear_planner": "mlp_planner",
        "mlp_planner": "mlp_planner",
        "transformer_planner": "transformer_planner",
        "vit_planner": "vit_planner",
    }
    
    if model_name not in model_name_map:
        raise ValueError(f"Unknown model_name: {model_name}. Use one of: {list(model_name_map.keys())}")
    
    model_key = model_name_map[model_name]

    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Determine transform pipeline based on model if not explicitly set
    if transform_pipeline is None:
        if model_key == "vit_planner":
            transform_pipeline = "default"  # ViT needs images
        else:
            transform_pipeline = "state_only"  # Others only need track data

    # Load data
    print("Loading data...")
    train_loader = load_data(
        dataset_path="./drive_data/train",
        transform_pipeline=transform_pipeline,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = load_data(
        dataset_path="./drive_data/val",
        transform_pipeline=transform_pipeline,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Create model
    print(f"Creating {model_key} model...")
    model = MODEL_FACTORY[model_key]()
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr)

    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(num_epoch):
        print(f"\nEpoch {epoch+1}/{num_epoch}")
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train L1: {train_metrics['l1_error']:.4f}, "
              f"Train Long: {train_metrics['longitudinal_error']:.4f}, Train Lat: {train_metrics['lateral_error']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val L1: {val_metrics['l1_error']:.4f}, "
              f"Val Long: {val_metrics['longitudinal_error']:.4f}, Val Lat: {val_metrics['lateral_error']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = save_model(model)
            print(f"Saved best model to {save_path}")

    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Train a planner model")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["mlp_planner", "transformer_planner", "vit_planner"],
                       help="Model to train")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    # Determine transform pipeline
    if args.model == "vit_planner":
        transform_pipeline = "default"  # Needs images
    else:
        transform_pipeline = "state_only"  # Only needs track data
    
    # Call the train function
    train(
        model_name=args.model,
        transform_pipeline=transform_pipeline,
        num_workers=args.num_workers,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epoch=args.epochs,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
