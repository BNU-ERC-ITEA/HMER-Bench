import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src" / "postprocess"), str(ROOT / "src" / "eval")]

from evaluate import evaluate
from postprocess_sasr import (
    build_dictionary,
    extract_prediction,
    extract_results,
    tokenize_results,
)


class SasrPipelineTest(unittest.TestCase):
    def test_extracts_nested_final_box(self):
        answer = r"analysis \boxed{\frac{1}{2}} trailing \boxed{x_{1}}"
        self.assertEqual(extract_prediction(answer), r"x_{1}")

    def test_postprocesses_and_evaluates_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_category = root / "predictions" / "SSE"
            prediction_category.mkdir(parents=True)
            (prediction_category / "origin_answer.json").write_text(
                json.dumps([{
                    "image_name": "SSE_000001",
                    "original_answer": r"\boxed{\frac{1}{2}}",
                }]),
                encoding="utf-8",
            )
            result_files = extract_results(root / "predictions", "origin_answer.json", "result.txt")
            vocabulary = build_dictionary(result_files, root / "predictions" / "dictionary.txt")
            tokenize_results(result_files, vocabulary, "labels.txt")

            truth_category = root / "ground_truth" / "SSE"
            truth_category.mkdir(parents=True)
            (truth_category / "labels_space.txt").write_text(
                r"SSE_000001	$ \frac { 1 } { 2 } $" + "\n", encoding="utf-8"
            )
            report = evaluate(
                root / "ground_truth",
                root / "predictions",
                "labels_space.txt",
                "labels.txt",
            )
            self.assertIn("ExpRate: 1/1 (100.00%)", report)


if __name__ == "__main__":
    unittest.main()
