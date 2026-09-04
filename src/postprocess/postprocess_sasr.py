"""Convert raw SASR responses into the tokenized labels used for ExpRate."""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Set


def extract_prediction(text: Optional[str]) -> str:
    """Return the final boxed expression, matching the original SASR release."""
    if not text:
        return ""
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    matches = list(re.finditer(r"\\boxed\{", text))
    if matches:
        start = matches[-1].end()
        depth = 1
        end = start
        while end < len(text) and depth:
            if text[end] == "{" and (end == 0 or text[end - 1] != "\\"):
                depth += 1
            elif text[end] == "}" and (end == 0 or text[end - 1] != "\\"):
                depth -= 1
            end += 1
        return text[start : end - 1]
    displays = list(re.finditer(r"\$\$(.*?)\$\$", text))
    if displays:
        return displays[-1].group(1)
    inline = list(re.finditer(r"\$(.*?)\$", text))
    return inline[-1].group(1) if inline else text


def extract_tokens(text: str) -> List[str]:
    """Use the original release's dictionary-token extraction rules."""
    tokens, index = [], 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            for command, offset in (("\\begin", 6), ("\\end", 4)):
                if text.startswith(command + "{", index):
                    close = text.find("}", index + offset + 1)
                    name = text[index + offset + 1 : close]
                    if close != -1 and re.fullmatch(r"[a-zA-Z0-9*_-]+", name):
                        tokens.append(text[index : close + 1])
                        index = close + 1
                        break
            else:
                if text[index + 1] == "\\":
                    tokens.append("\\\\")
                    index += 2
                elif text[index + 1].isascii() and text[index + 1].isalpha():
                    token, index = "\\" + text[index + 1], index + 2
                    while index < len(text) and text[index].isascii() and text[index].isalpha():
                        if token[-1].islower() and text[index].isupper():
                            break
                        token += text[index]
                        index += 1
                    tokens.append(token)
                else:
                    tokens.append(text[index : index + 2])
                    index += 2
                continue
            continue
        if char == "\\":
            tokens.append(char)
        elif not char.isspace():
            tokens.append(char)
        index += 1
    return tokens


def tokenize(text: str, vocabulary: Set[str]) -> List[str]:
    """Longest-match tokenization with the original 20-character window."""
    tokens, index = [], 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        for length in range(min(20, len(text) - index), 0, -1):
            candidate = text[index : index + length]
            if candidate in vocabulary:
                tokens.append(candidate)
                index += length
                break
        else:
            tokens.append(text[index])
            index += 1
    return tokens


def category_dirs(root: Path) -> List[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir())


def extract_results(prediction_root: Path, raw_name: str, result_name: str) -> List[Path]:
    result_files = []
    for category in category_dirs(prediction_root):
        raw_path = category / raw_name
        if not raw_path.is_file():
            continue
        records = json.loads(raw_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list: {raw_path}")
        lines = []
        for record in records:
            if not isinstance(record, dict) or "image_name" not in record:
                raise ValueError(f"Invalid prediction record in {raw_path}: {record!r}")
            lines.append(f"{record['image_name']}\t{extract_prediction(record.get('original_answer'))}")
        result_path = category / result_name
        result_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        result_files.append(result_path)
    if not result_files:
        raise FileNotFoundError(f"No {raw_name} files found below {prediction_root}")
    return result_files


def build_dictionary(result_files: List[Path], dictionary_path: Path) -> Set[str]:
    vocabulary = set()
    for result_path in result_files:
        for line in result_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                vocabulary.update(extract_tokens(parts[1]))
    dictionary_path.write_text("\n".join(sorted(vocabulary)) + "\n", encoding="utf-8")
    return vocabulary


def tokenize_results(result_files: List[Path], vocabulary: Set[str], labels_name: str) -> None:
    for result_path in result_files:
        lines = []
        for line in result_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                lines.append(f"{parts[0]}\t{' '.join(tokenize(parts[1], vocabulary))}")
            else:
                lines.append(line)
        (result_path.parent / labels_name).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--raw-name", default="origin_answer.json")
    parser.add_argument("--result-name", default="result.txt")
    parser.add_argument("--labels-name", default="labels.txt")
    parser.add_argument("--dictionary", type=Path)
    args = parser.parse_args()

    result_files = extract_results(args.prediction_root, args.raw_name, args.result_name)
    dictionary_path = args.dictionary or args.prediction_root / "dictionary.txt"
    vocabulary = build_dictionary(result_files, dictionary_path)
    tokenize_results(result_files, vocabulary, args.labels_name)
    print(f"Postprocessed {len(result_files)} categories; {len(vocabulary)} dictionary tokens.")


if __name__ == "__main__":
    main()
