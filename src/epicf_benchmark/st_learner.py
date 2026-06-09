from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from epicf_benchmark.data import FluDataset
from epicf_benchmark.models import FluBiRNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_array_both(array: np.ndarray, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{prefix}.npy", array)
    pd.DataFrame(array.reshape(-1, 1)).to_csv(
        output_dir / f"{prefix}.csv", header=False, index=False
    )


def build_loader(
    y_file: Path,
    a_file: Path,
    x_file: Path,
    indices: list[int],
    *,
    batch_size: int,
    shuffle: bool,
    policy_dim: int,
    num_units: int,
    sequence_length: int,
    num_covariates: int,
) -> DataLoader:
    dataset = FluDataset(
        y_file,
        a_file,
        x_file,
        indices,
        policy_dim=policy_dim,
        num_units=num_units,
        sequence_length=sequence_length,
        num_covariates=num_covariates,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def sample_treatment_mean(x: torch.Tensor, treatment_col: int = 0) -> torch.Tensor:
    return x[:, :, treatment_col].mean(dim=1)


def train_epoch_s(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def predict_s(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in dataloader:
            preds.append(model(x.to(device)).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_epoch_t(
    model0: torch.nn.Module,
    model1: torch.nn.Module,
    dataloader: DataLoader,
    opt0: torch.optim.Optimizer,
    opt1: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    treatment_col: int,
) -> tuple[float, float]:
    model0.train()
    model1.train()
    total_loss0 = 0.0
    total_loss1 = 0.0
    n0 = 0
    n1 = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        mean_a = sample_treatment_mean(x, treatment_col)
        mask0 = mean_a <= 0.5
        mask1 = mean_a > 0.5

        if mask0.any():
            opt0.zero_grad()
            loss0 = criterion(model0(x[mask0]), y[mask0])
            loss0.backward()
            opt0.step()
            total_loss0 += loss0.item()
            n0 += 1

        if mask1.any():
            opt1.zero_grad()
            loss1 = criterion(model1(x[mask1]), y[mask1])
            loss1.backward()
            opt1.step()
            total_loss1 += loss1.item()
            n1 += 1

    return total_loss0 / max(1, n0), total_loss1 / max(1, n1)


def predict_t(
    model0: torch.nn.Module,
    model1: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    treatment_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model0.eval()
    model1.eval()
    preds = []
    y0_all = []
    y1_all = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            mean_a = sample_treatment_mean(x, treatment_col)
            y0 = model0(x)
            y1 = model1(x)
            y_hat = torch.where(mean_a.unsqueeze(-1) > 0.5, y1, y0)
            preds.append(y_hat.detach().cpu().numpy())
            y0_all.append(y0.detach().cpu().numpy())
            y1_all.append(y1.detach().cpu().numpy())

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(y0_all, axis=0),
        np.concatenate(y1_all, axis=0),
    )


def run_st_learner(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_indices = list(range(args.train_units))
    test_indices = list(range(args.test_units))

    train_loader = build_loader(
        data_dir / args.train_y,
        data_dir / args.train_a,
        data_dir / args.train_x,
        train_indices,
        batch_size=args.batch_size,
        shuffle=True,
        policy_dim=args.policy_dim,
        num_units=args.train_units,
        sequence_length=args.sequence_length,
        num_covariates=args.num_covariates,
    )
    train_eval_loader = build_loader(
        data_dir / args.train_y,
        data_dir / args.train_a,
        data_dir / args.train_x,
        train_indices,
        batch_size=args.batch_size,
        shuffle=False,
        policy_dim=args.policy_dim,
        num_units=args.train_units,
        sequence_length=args.sequence_length,
        num_covariates=args.num_covariates,
    )
    test_loader = build_loader(
        data_dir / args.test_y,
        data_dir / args.test_a,
        data_dir / args.test_x,
        test_indices,
        batch_size=args.batch_size,
        shuffle=False,
        policy_dim=args.policy_dim,
        num_units=args.test_units,
        sequence_length=args.sequence_length,
        num_covariates=args.num_covariates,
    )

    first_x, _ = next(iter(train_loader))
    input_dim = first_x.shape[-1]
    print(f"Using device: {device}")
    print(f"Input dim: {input_dim}, treatment_col: {args.treatment_col}")

    criterion = nn.MSELoss()
    s_model = make_rnn(args, input_dim).to(device)
    s_opt = torch.optim.Adam(s_model.parameters(), lr=args.lr)

    s_best = copy.deepcopy(s_model.state_dict())
    s_best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        s_loss = train_epoch_s(s_model, train_loader, s_opt, criterion, device)
        if s_loss < s_best_loss:
            s_best_loss = s_loss
            s_best = copy.deepcopy(s_model.state_dict())
        print(f"[S] Epoch {epoch:03d} | train={s_loss:.6f}")

    s_model.load_state_dict(s_best)
    torch.save(s_model.state_dict(), model_dir / f"slearner_rnn_mean_{args.run_id}.pt")

    t_model0 = make_rnn(args, input_dim).to(device)
    t_model1 = make_rnn(args, input_dim).to(device)
    opt0 = torch.optim.Adam(t_model0.parameters(), lr=args.lr)
    opt1 = torch.optim.Adam(t_model1.parameters(), lr=args.lr)

    best0 = copy.deepcopy(t_model0.state_dict())
    best1 = copy.deepcopy(t_model1.state_dict())
    best0_loss = float("inf")
    best1_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss0, loss1 = train_epoch_t(
            t_model0, t_model1, train_loader, opt0, opt1, criterion, device, args.treatment_col
        )
        if loss0 < best0_loss:
            best0_loss = loss0
            best0 = copy.deepcopy(t_model0.state_dict())
        if loss1 < best1_loss:
            best1_loss = loss1
            best1 = copy.deepcopy(t_model1.state_dict())
        print(f"[T] Epoch {epoch:03d} | m0 train={loss0:.6f} | m1 train={loss1:.6f}")

    t_model0.load_state_dict(best0)
    t_model1.load_state_dict(best1)
    torch.save(t_model0.state_dict(), model_dir / f"tlearner_rnn_m0_mean_{args.run_id}.pt")
    torch.save(t_model1.state_dict(), model_dir / f"tlearner_rnn_m1_mean_{args.run_id}.pt")

    s_f = predict_s(s_model, train_eval_loader, device)
    s_cf = predict_s(s_model, test_loader, device)
    t_f, _, _ = predict_t(t_model0, t_model1, train_eval_loader, device, args.treatment_col)
    t_cf, t_y0, t_y1 = predict_t(t_model0, t_model1, test_loader, device, args.treatment_col)

    save_array_both(s_f, output_dir, f"slearner_rnn_mean_f_{args.run_id}")
    save_array_both(s_cf, output_dir, f"slearner_rnn_mean_cf_{args.run_id}")
    save_array_both(t_f, output_dir, f"tlearner_rnn_mean_f_{args.run_id}")
    save_array_both(t_cf, output_dir, f"tlearner_rnn_mean_cf_{args.run_id}")
    save_array_both(t_y0, output_dir, f"tlearner_rnn_mean_y0_{args.run_id}")
    save_array_both(t_y1, output_dir, f"tlearner_rnn_mean_y1_{args.run_id}")

    print("Done.")
    print(f"S factual shape: {s_f.shape}, S cf shape: {s_cf.shape}")
    print(f"T factual shape: {t_f.shape}, T cf shape: {t_cf.shape}")


def make_rnn(args: argparse.Namespace, input_dim: int) -> FluBiRNN:
    return FluBiRNN(
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type="lstm",
        dropout=args.dropout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mean-treatment RNN S/T learner baseline.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-y", default="y_train.csv")
    parser.add_argument("--train-a", default="a_train.csv")
    parser.add_argument("--train-x", default="x_train.csv")
    parser.add_argument("--test-y", default="y_test.csv")
    parser.add_argument("--test-a", default="a_test.csv")
    parser.add_argument("--test-x", default="x_test.csv")
    parser.add_argument("--train-units", type=int, default=158)
    parser.add_argument("--test-units", type=int, default=158)
    parser.add_argument("--policy-dim", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=168)
    parser.add_argument("--num-covariates", type=int, default=17)
    parser.add_argument("--treatment-col", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-id", default="1_new")
    parser.add_argument("--device", default=None)
    return parser
