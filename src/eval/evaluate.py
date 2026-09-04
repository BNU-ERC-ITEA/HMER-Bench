"""Evaluate tokenized SASR predictions with the original ExpRate protocol."""

import argparse
from pathlib import Path
from typing import Dict, List


def cal_distance(first: List[str], second: List[str]) -> int:
    """Levenshtein distance used by the original SASR evaluator."""
    if not first or not second:
        return len(first) + len(second)
    previous = list(range(len(second) + 1))
    for row, first_token in enumerate(first, 1):
        current = [row]
        for column, second_token in enumerate(second, 1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (first_token != second_token),
            ))
        previous = current
    return previous[-1]


def load_labels(root: Path, filename: str) -> Dict[str, Dict[str, List[str]]]:
    data = {}
    for category in sorted(path for path in root.iterdir() if path.is_dir()):
        label_path = category / filename
        if not label_path.is_file():
            continue
        labels = {}
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                labels[parts[0]] = parts[1].replace("$", "").split()
        data[category.name] = labels
    return data


def evaluate(gt_root: Path, prediction_root: Path, gt_name: str, prediction_name: str) -> str:
    ground_truth = load_labels(gt_root, gt_name)
    predictions = load_labels(prediction_root, prediction_name)
    lines = ["=" * 80, "OCR Evaluation Results", "=" * 80,
             f"GT Root: {gt_root}", f"Pred Root: {prediction_root}", ""]
    total_exact = total_accuracy = total_samples = 0
    for category in sorted(set(ground_truth) | set(predictions)):
        gt_labels = ground_truth.get(category, {})
        pred_labels = predictions.get(category, {})
        exact = accuracy = samples = 0
        for image_name in sorted(set(gt_labels) | set(pred_labels)):
            gt_tokens, pred_tokens = gt_labels.get(image_name, []), pred_labels.get(image_name, [])
            is_exact = gt_tokens == pred_tokens
            max_length = max(len(gt_tokens), len(pred_tokens))
            sample_accuracy = (1 - cal_distance(gt_tokens, pred_tokens) / max_length) * 100 if max_length else 100.0
            exact += is_exact
            accuracy += sample_accuracy
            samples += 1
        total_exact += exact
        total_accuracy += accuracy
        total_samples += samples
        lines.extend((
            f"{category}:",
            f"  Total Samples: {samples}",
            f"  ExpRate: {exact}/{samples} ({exact / samples * 100 if samples else 0:.2f}%)",
            f"  Accuracy (avg): {accuracy / samples if samples else 0:.2f}%", "",
        ))
    lines.extend((
        "=" * 80, "Overall Metrics:", f"  Total Samples: {total_samples}",
        f"  ExpRate: {total_exact}/{total_samples} ({total_exact / total_samples * 100 if total_samples else 0:.2f}%)",
        f"  Accuracy (avg): {total_accuracy / total_samples if total_samples else 0:.2f}%", "=" * 80,
    ))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--ground-truth-labels", default="labels_space.txt")
    parser.add_argument("--prediction-labels", default="labels.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.ground_truth_root, args.prediction_root,
                      args.ground_truth_labels, args.prediction_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
