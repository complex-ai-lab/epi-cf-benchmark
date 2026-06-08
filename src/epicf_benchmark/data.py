from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_SEQUENCE_LENGTH = 168
DEFAULT_NUM_COUNTIES = 158
DEFAULT_NUM_COVARIATES = 17


@dataclass(frozen=True)
class PolicyConfig:
    """Shape metadata for a benchmark policy setting."""

    name: str
    policy_dim: int
    num_units: int = DEFAULT_NUM_COUNTIES
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    num_covariates: int = DEFAULT_NUM_COVARIATES

    @property
    def input_dim(self) -> int:
        return self.policy_dim + self.num_covariates


SINGLE_POLICY = PolicyConfig(name="single_policy", policy_dim=1)
MULTI_POLICY = PolicyConfig(name="multi_policy", policy_dim=2)


class FluDataset(Dataset):
    """Dataset used by the benchmark ML baselines.

    The source CSV files were produced by the ABM/data preprocessing pipeline.
    Values are flattened in the files and reshaped into county-by-time tensors.
    """

    def __init__(
        self,
        y_file: Union[str, Path],
        policy_file: Union[str, Path],
        covariate_file: Union[str, Path],
        indices: Optional[Iterable[int]] = None,
        *,
        policy_dim: int,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        num_covariates: int = DEFAULT_NUM_COVARIATES,
        num_units: Optional[int] = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.policy_dim = policy_dim
        self.num_covariates = num_covariates

        y_values = _read_values(y_file, header=None)
        policy_values = _read_values(policy_file, header=None)
        covariate_values = _read_values(covariate_file, header=0)

        inferred_units = len(y_values) // sequence_length
        self.num_units = num_units or inferred_units

        self.y = y_values.reshape(self.num_units, sequence_length)
        self.policy = policy_values.reshape(self.num_units, sequence_length, policy_dim)
        self.covariates = covariate_values.reshape(
            self.num_units, sequence_length, num_covariates
        )

        selected_indices = list(range(self.num_units)) if indices is None else list(indices)
        self.inputs: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []

        for idx in selected_indices:
            x = np.concatenate([self.policy[idx], self.covariates[idx]], axis=1)
            self.inputs.append(torch.tensor(x, dtype=torch.float32))
            self.targets.append(torch.tensor(self.y[idx], dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def all_indices(num_units: int = DEFAULT_NUM_COUNTIES) -> list[int]:
    return list(range(num_units))


def _read_values(path: Union[str, Path], *, header: Optional[int]) -> np.ndarray:
    data = pd.read_csv(path, header=header).values
    return data.reshape(-1)
