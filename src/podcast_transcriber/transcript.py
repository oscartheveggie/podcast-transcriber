"""Transcript export helpers."""

from pathlib import Path


class TranscriptExporter:
    """Utilities for persisting transcript text."""

    @staticmethod
    def _export_to_txt(transcript: str, output_path: str | Path) -> Path:
        """Write transcript text to a UTF-8 encoded .txt file and return the path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(transcript, encoding="utf-8")
        return path


if __name__ == "__main__":
    exporter = TranscriptExporter()
    sample_transcript = "This is a sample transcript for testing."
    output_file = "output/sample_transcript.txt"
    saved_path = exporter._export_to_txt(sample_transcript, output_file)
