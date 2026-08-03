# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate the fixed-shape EP8 hash-routing AIV extern."""

import pypto.language as pl

from route_hash_aiv import (
    EXPERTS,
    SCORE_PAD,
    TOKENS,
    TOPK,
    route_hash_aiv_cce,
)


VOCAB = 64


@pl.jit
def route_hash_aiv_probe(
    route_scores: pl.Tensor[[TOKENS, SCORE_PAD], pl.FP32],
    input_ids: pl.Tensor[[TOKENS], pl.INT64],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    indices: pl.Out[pl.Tensor[[TOKENS, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[TOKENS, TOPK], pl.FP32]],
):
    with pl.spmd(
        TOKENS,
        name_hint="route_hash_aiv",
        allow_early_resolve=True,
    ):
        weights, indices = route_hash_aiv_cce(
            weights,
            indices,
            route_scores,
            input_ids,
            tid2eid,
            num_tokens,
        )
    return indices, weights


def build_specs(num_tokens):
    import torch
    from golden import ScalarSpec, TensorSpec

    def init_scores():
        base = torch.arange(
            TOKENS * SCORE_PAD, dtype=torch.float32
        ).reshape(TOKENS, SCORE_PAD)
        return base.remainder(EXPERTS).add_(1.0).div_(EXPERTS)

    def init_tid2eid():
        token = torch.arange(VOCAB, dtype=torch.int32).reshape(VOCAB, 1)
        slot = torch.arange(TOPK, dtype=torch.int32).reshape(1, TOPK)
        return (token * 7 + slot * 19).remainder(EXPERTS)

    return [
        TensorSpec(
            "route_scores",
            [TOKENS, SCORE_PAD],
            torch.float32,
            init_value=init_scores,
        ),
        TensorSpec(
            "input_ids",
            [TOKENS],
            torch.int64,
            init_value=lambda: torch.arange(TOKENS, dtype=torch.int64),
        ),
        TensorSpec(
            "tid2eid",
            [VOCAB, TOPK],
            torch.int32,
            init_value=init_tid2eid,
        ),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec(
            "indices", [TOKENS, TOPK], torch.int32, is_output=True
        ),
        TensorSpec(
            "weights", [TOKENS, TOPK], torch.float32, is_output=True
        ),
    ]


def golden(tensors):
    import torch

    num_tokens = max(
        0, min(TOKENS, int(tensors.get("num_tokens", TOKENS)))
    )
    indices = tensors["tid2eid"][
        tensors["input_ids"].to(torch.int64)
    ]
    scores = torch.gather(
        tensors["route_scores"][:, :EXPERTS],
        dim=-1,
        index=indices.to(torch.int64),
    )
    weights = scores / scores.sum(dim=-1, keepdim=True) * 1.5
    tensors["indices"].zero_()
    tensors["weights"].zero_()
    tensors["indices"][:num_tokens] = indices[:num_tokens]
    tensors["weights"][:num_tokens] = weights[:num_tokens]


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=TOKENS)
    args = parser.parse_args()

    result = run_jit(
        fn=route_hash_aiv_probe,
        specs=build_specs(args.num_tokens),
        golden_fn=golden,
        compile_cfg=dict(),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=1e-6,
        atol=1e-6,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
