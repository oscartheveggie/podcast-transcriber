import os
import re
from jiwer import wer

PROJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

TRANSCRIPT_SECTION_RE = re.compile(
    r"^===== Transcrip(?:t|tion) Output =====\s*$",
    re.MULTILINE,
)
MODEL_BLOCK_RE = re.compile(
    r"(?ms)^Model:\s*(?P<model>.+?)\s*\n"
    r"(?:Accuracy:\s*.*?(?:\n|(?=Transcript:)))?"
    r"Transcript:\s*\n"
    r"(?P<transcript>.*?)(?=^Model:\s*|\Z)"
)


def parse_model_transcripts(hypothesis_text):
    section_match = TRANSCRIPT_SECTION_RE.search(hypothesis_text)
    transcript_section = (
        hypothesis_text[section_match.end():] if section_match else hypothesis_text
    )

    return {
        match.group("model").strip(): match.group("transcript").strip()
        for match in MODEL_BLOCK_RE.finditer(transcript_section)
    }


def calculate_accuracy(hypothesis_path, reference_path):
    with open(reference_path, "r", encoding="utf-8") as reference_file:
        reference = reference_file.read().strip()

    with open(hypothesis_path, "r", encoding="utf-8") as hypothesis_file:
        hypothesis_text = hypothesis_file.read()

    model_transcripts = parse_model_transcripts(hypothesis_text)
    accuracies = {
        model: (1 - wer(reference, transcript)) * 100
        for model, transcript in model_transcripts.items()
    }

    def add_accuracy(match):
        model_line = match.group(1)
        model = match.group(2).strip()
        accuracy = accuracies.get(model)
        if accuracy is None:
            return match.group(0)

        return f"{model_line}\nAccuracy: {accuracy:.2f}\n"

    updated_hypothesis_text = re.sub(
        r"(?m)^(Model:\s*(.+?))[ \t]*\r?\n"
        r"(?:Accuracy:\s*[^\r\n]*?(?:\r?\n|(?=Transcript:)))?"
        r"(?=Transcript:)",
        add_accuracy,
        hypothesis_text,
    )

    with open(hypothesis_path, "w", encoding="utf-8") as hypothesis_file:
        hypothesis_file.write(updated_hypothesis_text)

    return accuracies


if __name__ == "__main__":

    reference_file = "cantonese"
    hypothesis_file = "test_results_2026-06-22_14-43-25"

    reference_path = os.path.join(PROJ_DIR, "input", f"{reference_file}.txt")
    hypothesis_path = os.path.join(
        PROJ_DIR,
        "output",
        f"{hypothesis_file}.txt",
    )
    calculate_accuracy(hypothesis_path, reference_path)
