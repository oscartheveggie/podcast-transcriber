import os
import re
import unicodedata
from jiwer import Compose, Strip, ToLowerCase
from opencc import OpenCC

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
EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U0000200D"
    "\U0000FE0F"
    "]+"
)
CJK_CHAR_RE = re.compile(
    "["
    "\U00003400-\U00004DBF"
    "\U00004E00-\U00009FFF"
    "\U0000F900-\U0000FAFF"
    "\U00020000-\U0002A6DF"
    "\U0002A700-\U0002B73F"
    "\U0002B740-\U0002B81F"
    "\U0002B820-\U0002CEAF"
    "\U0002CEB0-\U0002EBEF"
    "\U00030000-\U0003134F"
    "]"
)
MIXED_TOKEN_RE = re.compile(
    CJK_CHAR_RE.pattern
    + r"|[a-z0-9]+(?:'[a-z0-9]+)?"
    + r"|[^\s]",
    re.IGNORECASE,
)
SIMPLIFIED_TO_TRADITIONAL = OpenCC("s2t")


class RemoveEmojis:
    def __call__(self, text):
        if isinstance(text, list):
            return [self(sentence) for sentence in text]

        return EMOJI_RE.sub("", text)


class ConvertSimplifiedToTraditional:
    def __call__(self, text):
        if isinstance(text, list):
            return [self(sentence) for sentence in text]

        return SIMPLIFIED_TO_TRADITIONAL.convert(text)


class ReplacePunctuationWithSpace:
    def __call__(self, text):
        if isinstance(text, list):
            return [self(sentence) for sentence in text]

        return "".join(
            " " if unicodedata.category(character).startswith("P") else character
            for character in text
        )


class NormalizeWhitespace:
    def __call__(self, text):
        if isinstance(text, list):
            return [self(sentence) for sentence in text]

        return re.sub(r"\s+", " ", text)


class CombineSeparatedCharacters:
    def __call__(self, text):
        if isinstance(text, list):
            return [self(sentence) for sentence in text]

        words = text.split(" ")
        combined_words = []
        current_character_run = []

        for word in words:
            if len(word) == 1 and word.isalnum():
                current_character_run.append(word)
                continue

            if current_character_run:
                combined_words.append("".join(current_character_run))
                current_character_run = []

            combined_words.append(word)

        if current_character_run:
            combined_words.append("".join(current_character_run))

        return " ".join(combined_words)


PREPROCESS_TRANSFORM = Compose(
    [
        RemoveEmojis(),
        ToLowerCase(),
        ConvertSimplifiedToTraditional(),
        ReplacePunctuationWithSpace(),
        NormalizeWhitespace(),
        Strip(),
        CombineSeparatedCharacters(),
        NormalizeWhitespace(),
        Strip(),
    ]
)


def preprocess_text(text):
    return PREPROCESS_TRANSFORM(text)


def tokenize_mixed_text(text):
    return MIXED_TOKEN_RE.findall(preprocess_text(text))


def edit_distance(reference_tokens, hypothesis_tokens):
    previous_row = list(range(len(hypothesis_tokens) + 1))

    for reference_index, reference_token in enumerate(reference_tokens, start=1):
        current_row = [reference_index]

        for hypothesis_index, hypothesis_token in enumerate(hypothesis_tokens, start=1):
            substitution_cost = int(reference_token != hypothesis_token)
            current_row.append(
                min(
                    previous_row[hypothesis_index] + 1,
                    current_row[hypothesis_index - 1] + 1,
                    previous_row[hypothesis_index - 1] + substitution_cost,
                )
            )

        previous_row = current_row

    return previous_row[-1]


def mixed_error_rate(reference, hypothesis):
    reference_tokens = tokenize_mixed_text(reference)
    hypothesis_tokens = tokenize_mixed_text(hypothesis)

    if not reference_tokens:
        return 0.0 if not hypothesis_tokens else 1.0

    return edit_distance(reference_tokens, hypothesis_tokens) / len(reference_tokens)


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

    # Check whether file exists
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    if not os.path.exists(hypothesis_path):
        raise FileNotFoundError(f"Hypothesis file not found: {hypothesis_path}")

    with open(reference_path, "r", encoding="utf-8") as reference_file:
        reference = reference_file.read()

    with open(hypothesis_path, "r", encoding="utf-8") as hypothesis_file:
        hypothesis_text = hypothesis_file.read()

    model_transcripts = parse_model_transcripts(hypothesis_text)
    accuracies = {
        model: (1 - mixed_error_rate(reference, transcript)) * 100
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

    reference_file = input("Enter reference file name\n(without .txt, e.g., debug):\t\t\t")
    hypothesis_file = input("Enter hypothesis file name\n(without .txt, e.g., test_results_2026...):\t")

    reference_path = os.path.join(PROJ_DIR, "input", f"{reference_file}.txt")
    hypothesis_path = os.path.join(
        PROJ_DIR,
        "output",
        f"{hypothesis_file}.txt",
    )
    calculate_accuracy(hypothesis_path, reference_path)
