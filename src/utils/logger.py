"""
logger.py — Training metrics logger with CSV and console output.
"""

import os
import csv
import time
from typing import Dict, Any
from rich.console import Console
from rich.table import Table

console = Console()


class Logger:
    def __init__(self, log_dir: str, agent_name: str):
        self.log_dir    = log_dir
        self.agent_name = agent_name
        self.start_time = time.time()

        self._train_path = os.path.join(log_dir, "train_metrics.csv")
        self._eval_path  = os.path.join(log_dir, "eval_metrics.csv")

        self._train_file   = None
        self._train_writer = None
        self._all_train_keys = []   # grows as new metric keys appear

        self._eval_file   = None
        self._eval_writer = None

        self._eval_history = []
        self._pending_rows = []     # buffer rows until we know all keys

    def log_step(self, step: int, metrics: Dict[str, float]) -> None:
        row = {"step": step, **metrics}
        new_keys = [k for k in row if k not in self._all_train_keys]

        if new_keys:
            # New keys discovered — must rewrite the CSV with expanded header
            self._all_train_keys.extend(new_keys)
            self._pending_rows.append(row)

            # Close old file if open, rewrite with full header
            if self._train_file:
                self._train_file.close()

            self._train_file = open(self._train_path, "w", newline="")
            self._train_writer = csv.DictWriter(
                self._train_file,
                fieldnames=self._all_train_keys,
                extrasaction="ignore",
            )
            self._train_writer.writeheader()
            # Replay all buffered rows
            for r in self._pending_rows:
                self._train_writer.writerow(r)
            self._train_file.flush()
        else:
            self._pending_rows.append(row)
            self._train_writer.writerow(row)

    def log_eval(self, step: int, raw_score: float, normalized: float, info: Dict) -> None:
        elapsed = time.time() - self.start_time
        row = {
            "step": step,
            "raw_score": raw_score,
            "normalized_score": normalized,
            "elapsed_sec": elapsed,
            **{k: v for k, v in info.items() if isinstance(v, (int, float))},
        }
        if self._eval_writer is None:
            self._eval_file   = open(self._eval_path, "w", newline="")
            self._eval_writer = csv.DictWriter(
                self._eval_file,
                fieldnames=list(row.keys()),
                extrasaction="ignore",
            )
            self._eval_writer.writeheader()
        self._eval_writer.writerow(row)
        self._eval_file.flush()
        self._eval_history.append(row)

        console.print(
            f"[Step {step:>7,}] "
            f"score={normalized:6.1f} | raw={raw_score:8.1f} | "
            f"t={elapsed/60:.1f}min"
        )

    def summary(self) -> None:
        if self._train_file:
            self._train_file.close()
        if self._eval_file:
            self._eval_file.close()

        if not self._eval_history:
            return
        best = max(self._eval_history, key=lambda x: x["normalized_score"])
        table = Table(title=f"{self.agent_name.upper()} Training Summary")
        table.add_column("Metric"); table.add_column("Value")
        table.add_row("Best normalized score", f"{best['normalized_score']:.2f}")
        table.add_row("At step",               f"{best['step']:,}")
        table.add_row("Total time",            f"{(time.time()-self.start_time)/60:.1f} min")
        console.print(table)