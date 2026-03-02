#!/usr/bin/env python3
"""Lightweight xAI Batch monitor using a saved state JSON.

This tool is intentionally independent from heavy ML deps, so you can run it on
another machine/notebook just with requests installed.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import requests

DEFAULT_BASE_URL = "https://api.x.ai"


def load_state(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def get_json(url: str, headers_: dict, params: dict | None = None) -> dict:
    resp = requests.get(url, headers=headers_, params=params, timeout=120)
    resp.raise_for_status()
    if not resp.text:
        return {}
    return resp.json()


def cmd_status(args):
    state = load_state(args.state_file)
    batch_id = state.get("batch_id")
    if not batch_id:
        raise SystemExit(f"state file has no batch_id: {args.state_file}")

    api_key = args.xai_api_key or os.environ.get("XAI_API_KEY")
    if not api_key:
        raise SystemExit("XAI_API_KEY not set (or pass --xai_api_key)")

    data = get_json(f"{args.xai_api_base_url}/v1/batches/{batch_id}", headers(api_key))
    out = {
        "checked_at": datetime.utcnow().isoformat() + "Z",
        "batch_id": batch_id,
        "batch": data,
    }

    if args.save_snapshot:
        snap = args.save_snapshot
    else:
        snap = args.state_file + ".status.json"
    save_json(snap, out)

    counters = data.get("state", {})
    total = counters.get("num_requests", 0) or 0
    pending = counters.get("num_pending", 0) or 0
    success = counters.get("num_success", 0) or 0
    errors = counters.get("num_error", 0) or 0
    done = success + errors
    pct = (done / total * 100.0) if total else 0.0

    print(f"batch_id={batch_id}")
    print(f"total={total} pending={pending} success={success} error={errors} done={done} ({pct:.2f}%)")
    print(f"snapshot={snap}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="check batch progress using saved state file")
    status.add_argument("--state_file", required=True, help="path to .xai_batch_state_*.json")
    status.add_argument("--xai_api_key", default=None, help="xAI API key (or env XAI_API_KEY)")
    status.add_argument("--xai_api_base_url", default=DEFAULT_BASE_URL)
    status.add_argument("--save_snapshot", default=None, help="optional output json snapshot path")
    status.set_defaults(func=cmd_status)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        body = ""
        if e.response is not None:
            body = e.response.text[:1000]
        print(f"HTTP error: {e}\n{body}", file=sys.stderr)
        sys.exit(1)
