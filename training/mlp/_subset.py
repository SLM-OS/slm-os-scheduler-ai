"""SchedulerSubset — torch.utils.data.Subset with attribute forwarding.

`train_mlp` reads `.n_features` / `.n_actions` off the dataset to size
the model. A plain `torch.utils.data.Subset` (what `random_split`
returns) doesn't expose those attributes, so the fine-tune flow's
disjoint train/val split (#879) needed a wrapper that forwards them
from the parent SchedulerDataset. Lives in `training/mlp/` so the
class sits near the trainer it serves; importable from tests so the
property-forwarding contract has direct coverage independent of the
@slow end-to-end subprocess test.
"""

from __future__ import annotations

import torch.utils.data


class SchedulerSubset(torch.utils.data.Subset):
    """torch.utils.data.Subset that forwards `n_features` and
    `n_actions` from the parent SchedulerDataset. Required so a slice
    of a SchedulerDataset (e.g., from `random_split`) still type-checks
    at `train_mlp`'s call sites without changing the trainer's API.

    The parent must be a SchedulerDataset (or anything else that
    exposes both attributes); nesting another SchedulerSubset as the
    parent works too because each layer forwards.
    """

    @property
    def n_features(self):
        return self.dataset.n_features

    @property
    def n_actions(self):
        return self.dataset.n_actions
