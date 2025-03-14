# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention".

Modified from: https://github.com/idiap/fast-
transformers/blob/master/fast_transformers/attention/linear_attention.py.
"""

from typing import Optional

import torch
from torch.nn import Dropout

from kornia.core import Module, Tensor


def elu_feature_map(x: Tensor) -> Tensor:
    """Apply elu activation."""
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        q_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Multi-Head linear attention proposed in "Transformers are RNNs".

        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]

        Returns:
            queried_values: (N, L, H, D)

        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(Module):
    def __init__(self, use_dropout: bool = False, attention_dropout: float = 0.1) -> None:
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        q_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Multi-head scaled dot-product attention, a.k.a full attention.

        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]

        Returns:
            queried_values: (N, L, H, D)

        """
        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None and q_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float("-inf"))

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.size(3) ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()
