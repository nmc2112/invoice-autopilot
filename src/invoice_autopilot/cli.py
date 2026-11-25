"""Command line for the MVP.

  python -m invoice_autopilot.cli seed --n 80          # load labelled history
  python -m invoice_autopilot.cli run data/invoices.jsonl
  python -m invoice_autopilot.cli metrics
  python -m invoice_autopilot.cli export --out data/export_fec.txt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .classify import example_text
from .factory import build_autopilot
from .synth import generate_invoices


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    seed = sub.add_parser("seed", help="load labelled history into the embedding index")
    seed.add_argument("--n", type=int, default=80)
    run = sub.add_parser("run", help="process invoices from a JSONL file")
    run.add_argument("path")
    sub.add_parser("metrics")
    exp = sub.add_parser("export")
    exp.add_argument("--out", default="data/export_fec.txt")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    ap = build_autopilot()
    if args.cmd == "seed":
        history = [i for i in generate_invoices(args.n, seed=999) if i.expected_issue is None]
        ap.seed_history([(example_text(i.supplier, i.description), i.account) for i in history])
        print(f"Index now holds {len(ap.classifier)} labelled examples")
    elif args.cmd == "run":
        for line in Path(args.path).read_text(encoding="utf-8").splitlines():
            inv = json.loads(line)
            d = ap.process(inv["invoice_id"], inv["text"])
            print(f"{d.invoice_id}: {d.status:13s} account={d.predicted_account} conf={d.confidence} {d.issues}")
        print(json.dumps(ap.metrics(), indent=2))
    elif args.cmd == "metrics":
        print(json.dumps(ap.metrics(), indent=2))
    elif args.cmd == "export":
        Path(args.out).write_text(ap.export_fec(), encoding="utf-8")
        print(f"FEC written to {args.out}")


if __name__ == "__main__":
    main()
