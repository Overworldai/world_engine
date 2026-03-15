import argparse
import json

import torch


def _metrics(a: torch.Tensor, b: torch.Tensor):
    av = a.flatten()
    bv = b.flatten()
    d = (av - bv).abs()
    return {
        "cos": float(torch.nn.functional.cosine_similarity(av, bv, dim=0)),
        "mae": float(d.mean()),
        "rmse": float(torch.sqrt(((av - bv) ** 2).mean())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lhs", required=True, help="Path to lhs latents .pt")
    parser.add_argument("--rhs", required=True, help="Path to rhs latents .pt")
    parser.add_argument("--out", required=True, help="Path to output json report")
    args = parser.parse_args()

    lhs = torch.load(args.lhs, map_location="cpu")
    rhs = torch.load(args.rhs, map_location="cpu")
    n = min(len(lhs), len(rhs))
    per_step = {}
    worst = {"step": None, "cos": 10.0}
    for i in range(n):
        m = _metrics(lhs[i], rhs[i])
        per_step[f"step_{i+1:02d}"] = m
        if m["cos"] < worst["cos"]:
            worst = {"step": i + 1, "cos": m["cos"]}
    out = {
        "lhs": args.lhs,
        "rhs": args.rhs,
        "lhs_steps": len(lhs),
        "rhs_steps": len(rhs),
        "compared_steps": n,
        "worst_step_by_cos": worst,
        "per_step": per_step,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

