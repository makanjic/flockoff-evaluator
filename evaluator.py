import os
import argparse
import asyncio
import logging
import json
from typing import List, Tuple

import torch  # kept in case train_lora / HF stack expects it
import random

from flockoff.constants import Competition
from flockoff import constants
from flockoff.validator.validator_utils import compute_score
from flockoff.validator.trainer import train_lora


# HF cache directory
cache_dir = os.path.realpath(os.path.expanduser("~/data/hf_cache"))


async def evaluation_step(
    eval_data_dir: str,
    data_dir: str,
    hotkey_list: List[str],
    lucky_num: int,
) -> Tuple[List[float], List[float]]:
    """
    Run LoRA training for each miner (hotkey) and compute normalized scores.

    Args:
        eval_data_dir: Directory containing evaluation dataset.
        data_dir: Root directory containing miner subdirs (one per hotkey).
        hotkey_list: List of hotkeys (subdirectory names under data_dir).

    Returns:
        (raw_scores_this_epoch, normalized_scores_this_epoch)
    """
    logging.info("Starting run evaluation step")
    logging.info(f"Using evaluation directory: {eval_data_dir}")
    logging.info(f"Using data directory root: {data_dir}")

    competition = Competition.from_defaults()
    raw_scores_this_epoch: List[float] = []

    # ---- LoRA training / raw score collection ----
    for hotkey in hotkey_list:
        miner_data_dir = os.path.join(data_dir, hotkey)
        logging.info(f"[{hotkey}] Using data directory: {miner_data_dir}")

        if not os.path.isdir(miner_data_dir):
            logging.warning(f"[{hotkey}] Data directory does not exist: {miner_data_dir}")
            raw_scores_this_epoch.append(constants.DEFAULT_RAW_SCORE)
            continue

        try:
            logging.info(f"[{hotkey}] Starting LoRA training")
            eval_loss = train_lora(
                lucky_num,
                competition.bench,
                competition.rows,
                cache_dir=cache_dir,
                data_dir=miner_data_dir,
                eval_data_dir=eval_data_dir,
            )
            logging.info(f"[{hotkey}] Training complete with eval loss: {eval_loss}")
            raw_scores_this_epoch.append(eval_loss)

        except Exception as e:
            logging.error(f"[{hotkey}] train_lora error: {e}")
            if "CUDA" in str(e):
                logging.error("[FATAL] CUDA error detected, terminating process")
                os._exit(1)  # keep same behavior as original
            raw_scores_this_epoch.append(constants.DEFAULT_RAW_SCORE)

    # ---- Normalization ----
    logging.info("Normalizing raw scores")
    normalized_scores_this_epoch: List[float] = []

    for hotkey, current_raw_score in zip(hotkey_list, raw_scores_this_epoch):
        logging.debug(
            f"[{hotkey}] Computing normalized score with raw score {current_raw_score}"
        )

        if competition.bench is None or competition.bench <= 0:
            logging.warning(
                f"[{hotkey}] Invalid benchmark ({competition.bench}); "
                f"defaulting normalized score to {constants.DEFAULT_NORMALIZED_SCORE}"
            )
            normalized_score = constants.DEFAULT_NORMALIZED_SCORE
        else:
            # compute_score takes 8th argument; 7th and 8th are both competition.id
            normalized_score = compute_score(
                current_raw_score,
                competition.bench,
                competition.minb,
                competition.maxb,
                competition.pow,
                competition.bheight,
                competition.id,  # 7th argument
                competition.id,  # 8th argument
            )

        normalized_scores_this_epoch.append(normalized_score)

    logging.debug(f"Raw scores for this epoch: {raw_scores_this_epoch}")
    logging.debug(f"Normalized scores for this epoch: {normalized_scores_this_epoch}")

    return raw_scores_this_epoch, normalized_scores_this_epoch


def discover_hotkeys(data_dir: str) -> List[str]:
    """
    Discover hotkeys as subdirectory names under data_dir.
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")

    hotkeys = []
    for entry in os.listdir(data_dir):
        full_path = os.path.join(data_dir, entry)
        if os.path.isdir(full_path):
            hotkeys.append(entry)

    hotkeys.sort()
    return hotkeys


def load_hotkeys_from_file(path: str) -> List[str]:
    hotkeys: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hotkeys.append(line)
    return hotkeys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoRA evaluation for miner datasets and normalize scores."
    )
    parser.add_argument(
        "--eval-data-dir",
        type=str,
        required=True,
        help="Directory containing evaluation dataset.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing miner subdirectories (one per hotkey).",
    )
    parser.add_argument(
        "--hotkeys-file",
        type=str,
        default=None,
        help="Optional file with one hotkey per line. "
             "If not provided, hotkeys are inferred from subdirectories of data-dir.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="none",
        choices=["none", "hotkey", "raw_score", "norm_score"],
        help="Sort results by 'hotkey', 'raw_score', 'norm_score', or 'none' (no sorting).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="tsv",
        choices=["tsv", "csv", "json"],
        help="Output format for stdout and optional file: tsv (default), csv, or json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="If specified, save the output to this file.",
    )
    parser.add_argument(
        "--lucky-num",
        type=int,
        default=None,
        help=(
            "Seed value used in train_lora."
            "If omitted, a random 32-bit value is generated."
        ),
    )
    return parser.parse_args()


def format_results_line(result: dict, fmt: str) -> str:
    if fmt == "csv":
        return f"{result['hotkey']},{result['raw_score']:.6f},{result['norm_score']:.6f}"
    # default for line-based formats (tsv)
    return f"{result['hotkey']}\t{result['raw_score']:.6f}\t{result['norm_score']:.6f}"


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # lucky_num handling (your requirement)
    if args.lucky_num is None:
        lucky_num = int.from_bytes(os.urandom(4), "little")
        logging.info(f"Generated lucky_num={lucky_num}")
    else:
        lucky_num = args.lucky_num
        logging.info(f"Using provided lucky_num={lucky_num}")

    if args.hotkeys_file:
        logging.info(f"Loading hotkeys from file: {args.hotkeys_file}")
        hotkey_list = load_hotkeys_from_file(args.hotkeys_file)
    else:
        logging.info(f"Discovering hotkeys from data directory: {args.data_dir}")
        hotkey_list = discover_hotkeys(args.data_dir)

    logging.info(f"Found {len(hotkey_list)} hotkeys")

    raw_scores, normalized_scores = asyncio.run(
        evaluation_step(args.eval_data_dir, args.data_dir, hotkey_list, lucky_num)
    )

    # Build structured results
    results = [
        {
            "hotkey": hotkey,
            "raw_score": raw_score,
            "norm_score": norm_score,
        }
        for hotkey, raw_score, norm_score in zip(hotkey_list, raw_scores, normalized_scores)
    ]

    # Sorting (skip if 'none')
    if args.sort == "hotkey":
        results.sort(key=lambda x: x["hotkey"])
    elif args.sort == "raw_score":
        results.sort(key=lambda x: x["raw_score"])
    elif args.sort == "norm_score":
        results.sort(key=lambda x: x["norm_score"])

    # ---- Output to stdout ----
    if args.format == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(format_results_line(r, args.format))

    # ---- Save to file if requested ----
    if args.output:
        if args.format == "json":
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(format_results_line(r, args.format) + "\n")
        logging.info(f"Results written to {args.output}")
