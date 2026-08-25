#!/usr/bin/env python3
"""Average the tensors of N ink9 checkpoints into one soup checkpoint.

Usage: make_soup.py out.pth in1.pth in2.pth [...]
Keeps the non-tensor payload (config etc.) of the FIRST input.
"""
import sys

import torch


def main():
    out, first, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    payload = torch.load(first, map_location="cpu", weights_only=False)
    key = next(k for k in ("model", "model_state_dict", "state_dict", "network_weights")
               if k in payload) if isinstance(payload, dict) else None
    sd = payload[key] if key else payload
    acc = {k: v.clone().double() if torch.is_tensor(v) and v.is_floating_point() else v
           for k, v in sd.items()}
    n = 1
    for p in rest:
        other = torch.load(p, map_location="cpu", weights_only=False)
        osd = other[key] if key else other
        for k, v in osd.items():
            if torch.is_tensor(v) and v.is_floating_point():
                acc[k] += v.double()
        n += 1
    for k, v in acc.items():
        if torch.is_tensor(v) and v.is_floating_point():
            acc[k] = (v / n).to(sd[k].dtype)
    if key:
        payload[key] = acc
        torch.save(payload, out)
    else:
        torch.save(acc, out)
    print(f"soup of {n} -> {out} (state key: {key})")


if __name__ == "__main__":
    main()
