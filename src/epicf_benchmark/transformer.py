from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from epicf_benchmark.data import FluDataset, all_indices
from epicf_benchmark.models import FluTransformer


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            total_loss += criterion(model(x), y).item()
    return total_loss / len(dataloader)


def predict(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in dataloader:
            preds.append(model(x.to(device)).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_transformer(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = all_indices(args.num_units)
    train_set = _dataset(data_dir, args, "train", indices)
    val_set = _dataset(data_dir, args, "train", indices)
    test_set = _dataset(data_dir, args, "train", indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = FluTransformer(input_dim=args.policy_dim + args.num_covariates).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    model_path = output_dir / args.model_name
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} best_val_loss={best_val_loss:.6f}"
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss = eval_epoch(model, test_loader, criterion, device)
    print(f"Saved model: {model_path}")
    print(f"Test loss: {test_loss:.6f}")


def infer_transformer(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = all_indices(args.num_units)
    dataset = _dataset(data_dir, args, args.split, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = FluTransformer(input_dim=args.policy_dim + args.num_covariates).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    y_pred = predict(model, loader, device)

    npy_path = output_dir / f"{args.prefix}.npy"
    csv_path = output_dir / f"{args.prefix}.csv"
    np.save(npy_path, y_pred)
    pd.DataFrame(y_pred.reshape(-1, 1)).to_csv(csv_path, header=False, index=False)
    print(f"Prediction shape: {y_pred.shape}")
    print(f"Saved: {npy_path}")
    print(f"Saved: {csv_path}")


def _dataset(
    data_dir: Path,
    args: argparse.Namespace,
    split: str,
    indices: list[int],
) -> FluDataset:
    return FluDataset(
        data_dir / f"y_{split}.csv",
        data_dir / f"a_{split}.csv",
        data_dir / f"x_{split}.csv",
        indices,
        policy_dim=args.policy_dim,
        num_units=args.num_units,
        sequence_length=args.sequence_length,
        num_covariates=args.num_covariates,
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-dim", type=int, default=1)
    parser.add_argument("--num-units", type=int, default=158)
    parser.add_argument("--sequence-length", type=int, default=168)
    parser.add_argument("--num-covariates", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Transformer baseline.")
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-name", default="transformer.pt")
    return parser


def build_infer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Transformer inference.")
    add_common_args(parser)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--prefix", default="transformer_cf")
    return parser
