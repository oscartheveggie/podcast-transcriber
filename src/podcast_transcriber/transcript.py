"""Transcript helpers."""

from pathlib import Path
import whisperx
import gc
from whisperx.diarize import DiarizationPipeline
import torch
from tqdm import tqdm
import os
import dotenv

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") # get from https://huggingface.co/settings/tokens

SCRIPT_DIR = Path(__file__).parent

AVAILABLE_MODELS = ["whisperx_small", "whisperx_medium"] # for now, only whisperx is supported

class Transcriber:

    def __init__(self, model_name: str = "whisperx_small", 
                 model_dir: str = None, 
                 device: str = "cpu", 
                 compute_type: str = "int8",
                 batch_size: int = 16,
                 diarize: bool = False):
        
        self.model_name: str = model_name
        if self.model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Model '{self.model_name}' not supported. Available models: {AVAILABLE_MODELS}")
        
        self.model_dir: str = model_dir
        self.device: str = device
        self.compute_type: str = compute_type
        self.batch_size: int = batch_size
        self.diarize: bool = diarize
        self.transcript: dict | None = None

        self.model = None
    

    def _transcribe_whisperx(self, model_size: str, audio_path: str | Path, min_speakers: int = None, max_speakers: int = None) -> dict:
        """Transcribe the given audio file and return the transcript text."""
        
        self.model: whisperx.WhisperX = whisperx.load_model(
            model_size, 
            self.device, 
            compute_type=self.compute_type, 
            download_root=self.model_dir)

        # 1. Transcribe with whisperx
        print("\n" + "="*50 + "\nTranscribing audio with WhisperX...")
        audio = whisperx.load_audio(str(audio_path))
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        print(result["segments"]) # before alignment

        # 2. Align whisper output
        print("\n" + "="*50 + "\nAligning whisper output...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
        print(result["segments"]) # after alignment

        # 3. Assign speaker labels
        if (self.diarize):
            print("\n" + "="*50 + "\nAssigning speaker labels...")
            diarize_model = DiarizationPipeline(token=HF_TOKEN, device=self.device)

            # add min/max number of speakers if known
            diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

            result = whisperx.assign_word_speakers(diarize_segments, result)
            print("\n" + "="*50 + "\nDiarized segments:")
            print(diarize_segments)
            print("\n" + "="*50 + "\nSegments with speaker IDs:")
            print(result["segments"]) # segments are now assigned speaker IDs

        self.transcript = result

        return self.transcript
    

    def transcribe(self, audio_path: str | Path, min_speakers: int = None, max_speakers: int = None) -> dict:

        # 0. Check whether audio path exists
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # 1. WhisperX model
        if self.model_name.startswith("whisperx"):
            model_size = self.model_name.split("_")[1] # e.g. "small" or "medium"
            return self._transcribe_whisperx(model_size=model_size, 
                                             audio_path=audio_path, 
                                             min_speakers=min_speakers, 
                                             max_speakers=max_speakers)
        

    def export_to_txt(self, output_path: str | Path) -> str:
        """Export the transcript text to a .txt file and return the file path."""
        
        print("\nExporting transcript to text file...")
        if self.transcript is None:
            raise ValueError("No transcript available. Please transcribe an audio file first.") 
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            print("Writing transcript to file...")
            for segment in tqdm(self.transcript["segments"], desc="Exporting segments"):
                start_time = segment["start"]
                end_time = segment["end"]
                speaker_id = segment.get("speaker", "Unknown")
                text = segment["text"].strip()
                f.write(f"[{start_time:.2f} - {end_time:.2f}] Speaker {speaker_id}: {text}\n")

        return str(output_path)


if __name__ == "__main__":
    
    transcriber = Transcriber(model_dir=SCRIPT_DIR/"../../model")
    transcriber.transcribe(audio_path=SCRIPT_DIR / "../../input/transcoded.mp3", min_speakers=1, max_speakers=2)
    sample_transcript = "This is a sample transcript for testing."
    output_file = SCRIPT_DIR / "../../output/sample_transcript.txt"
    saved_path = transcriber.export_to_txt(output_file)
    