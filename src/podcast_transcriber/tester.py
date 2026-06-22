"""
File: tester.py
Description: Tests different transcription models and diarization methods to determine which is best for our use case.

Compares:
- Processing speed
"""

import time
import transcript
from transcript import Transcriber
from tqdm import tqdm
from mutagen.mp3 import MP3
from datetime import datetime
from plyer import notification


class Tester:

    def __init__(self, 
                 audio_path: str, 
                 models: list[str], 
                 model_dir: str,
                 output_dir: str,
                 device: str = "cpu",
                 compute_type: str = "int8",
                 batch_size: int = 16,
                 diarize: bool = False,
                 save_output: bool = True,
                 chi_sim: bool = False          # True for simplified Chinese, False for traditional Chinese.
                 ):

        if models is None or len(models) == 0:
            raise ValueError("At least one model must be specified for testing.")
        if not set(models).issubset(set(transcript.AVAILABLE_MODELS)):
            raise ValueError(f"One or more specified models are not supported. Available models: {transcript.AVAILABLE_MODELS}")

        self.audio_path: str = audio_path
        self.models: list[str] = models
        self.model_dir: str = model_dir
        self.save_output: bool = save_output
        self.output_dir: str = output_dir
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.diarize = diarize
        self.chi_sim = chi_sim
        self.results: dict[str, dict] = {} # model_name -> {transcript: str, processing_time: float}

    def _save_output(self):

        if not self.save_output:
            return

        with open(self.output_dir / f"test_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w", encoding="utf-8") as f:

            f.write(f"===== Model Test Results =====\n")
            f.write(f"Audio file: {self.audio_path} (duration: {self.get_audio_duration():.2f} seconds)\n")
            f.write(f"Models tested: {self.models}\n\n")

            for model_name, result in self.results.items():
                f.write(f"Model: {model_name}\n")
                f.write(f"Processing time: {result['processing_time']:.2f} seconds\n")
                f.write("\n")
            
            f.write(f"\n===== Transcript Output =====\n\n")
            
            for model_name, result in self.results.items():
                f.write(f"Model: {model_name}\n")
                f.write(f"Transcript:\n{result['transcript']}\n")
                f.write("\n")

    def run_tests(self):

        print("="*50)
        print("TRANSCRIPTION MODEL TESTER")
        print("="*50)
        
        print(f"\nAudio file: {self.audio_path} (duration: {self.get_audio_duration():.2f} seconds)")
        print(f"Models to test: {self.models}")
        
        for i, model_name in enumerate(self.models):
            print(f"\nTesting model: {model_name} ({i+1}/{len(self.models)})")
            transcriber = Transcriber(model_name=model_name, 
                                      model_dir=self.model_dir,
                                      device=self.device,
                                      compute_type=self.compute_type,
                                      batch_size=self.batch_size,
                                      diarize=self.diarize,
                                      chi_sim=self.chi_sim
                                      )
            start_time = time.time()
            transcript = transcriber.transcribe(self.audio_path)
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Processing time for {model_name}: {processing_time:.2f} seconds")

            transcript = transcript["text"] if "text" in transcript \
                else " ".join(segment["text"] for segment in transcript["segments"]) if "segments" in transcript and len(transcript["segments"]) > 0 \
                    else ""

            self.results[model_name] = {
                "transcript": transcript,
                "processing_time": processing_time
            }
        
        print(f"\nAll tests completed. Results:")
        print({model: {"processing_time": result["processing_time"]} for model, result in self.results.items()})

        self._save_output()
    
    def get_audio_duration(self) -> float:
        """Get the duration of the audio file in seconds."""
        audio = MP3(self.audio_path)
        return audio.info.length


if __name__ == "__main__":
    tester = Tester(
        audio_path="input/cantonese.mp3",
        models=["whisperx_small", "whisper_small", "sensevoice_small"],
        model_dir=transcript.SCRIPT_DIR / "../../model",
        output_dir=transcript.SCRIPT_DIR / "../../output",
        save_output=True
    )
    tester.run_tests()
    notification.notify(
        title="Transcription Model Tester",
        message="All transcription model tests have completed.",
        timeout=10
    )