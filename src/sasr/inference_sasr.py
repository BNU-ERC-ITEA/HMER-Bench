"""Qwen3-VL inference entry point for the standalone SASR paper release."""

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from tools_sasr import build_messages


RELEASE_ROOT = Path(__file__).resolve().parents[2]
PAPER_CATEGORIES = (
    "SSE",
    "LSE",
    "ACE",
    "SLD",
    "SAE",
    "RCE",
    "SOE",
    "CHEM",
    "CME",
    "VAE",
    "SDE",
    "MLD",
    "MAD",
)
VALID_MODES = ("SIL", "MSR", "SASR")


def normalize_mode(value: str) -> str:
    """Normalize a paper-method name to its canonical uppercase spelling."""
    mode = value.upper()
    if mode not in VALID_MODES:
        allowed = ", ".join(VALID_MODES)
        raise argparse.ArgumentTypeError(
            f"Unsupported mode: {value}. Choose one of: {allowed}."
        )
    return mode


def path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL inference for the SIL, MSR, or SASR paper method."
    )
    parser.add_argument(
        "--mode",
        type=normalize_mode,
        default=normalize_mode(os.environ.get("SASR_MODE", "SASR")),
        metavar="{SIL,MSR,SASR}",
        help="Paper method to run: SIL, MSR, or SASR.",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get(
            "SASR_MODEL_NAME", str(RELEASE_ROOT / "weight" / "Qwen3-VL-8B-Instruct")
        ),
        help="Model identifier served by the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SASR_BASE_URL", "http://127.0.0.1:22002/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SASR_API_KEY", "EMPTY"),
        help="API key for the inference endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("SASR_TIMEOUT", "60")),
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=path_from_env("SASR_INPUT_ROOT", RELEASE_ROOT / "data" / "hmerbench"),
        help="Directory containing category directories and their images subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=path_from_env("SASR_OUTPUT_ROOT", RELEASE_ROOT / "outputs"),
        help="Directory where raw inference outputs will be written.",
    )
    parser.add_argument(
        "--output-name",
        default=os.environ.get("SASR_OUTPUT_NAME"),
        help="Name of the run directory under output-root. Defaults to the selected mode.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=PAPER_CATEGORIES,
        default=list(PAPER_CATEGORIES),
        help="One or more paper category abbreviations to process. Defaults to all.",
    )
    return parser.parse_args()


def get_prediction(
    client: OpenAI, model_name: str, class_name: str, image_path: Path, mode: str
) -> str:
    messages = build_messages(class_name, str(image_path), mode=mode)
    completion = client.chat.completions.create(model=model_name, messages=messages)
    return completion.choices[0].message.content


def main() -> None:
    args = parse_args()
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    output_name = args.output_name or f"qwen3_vl_8b_sasr_{args.mode.lower()}"

    for class_name in args.classes:
        images_dir = args.input_root / class_name / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {images_dir}")

        output_dir = args.output_root / output_name / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        image_paths = sorted(path for path in images_dir.iterdir() if path.is_file())
        results = []

        for image_path in tqdm(image_paths, desc=f"Processing {class_name} ({args.mode})"):
            try:
                prediction = get_prediction(
                    client, args.model_name, class_name, image_path, args.mode
                )
            except Exception as error:
                prediction = f"ERROR: {error}"

            results.append(
                {
                    "image_name": image_path.stem,
                    "original_answer": prediction,
                }
            )

        origin_path = output_dir / "origin_answer.json"
        origin_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Processing complete. Results written to: {args.output_root / output_name}")


if __name__ == "__main__":
    main()
