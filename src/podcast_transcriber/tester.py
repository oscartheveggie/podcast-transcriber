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


class Tester:

    def __init__(self, 
                 audio_path: str, 
                 models: list[str], 
                 model_dir: str,
                 output_dir: str,
                 device: str = "cpu",
                 compute_size: str = "int8",
                 batch_size: int = 16,
                 diarize: bool = False,
                 save_output: bool = True
                 ):

        if models is None or len(models) == 0:
            raise ValueError("At least one model must be specified for testing.")
        if models not in transcript.AVAILABLE_MODELS:
            raise ValueError(f"One or more specified models are not supported. Available models: {transcript.AVAILABLE_MODELS}")

        self.audio_path: str = audio_path
        self.models: list[str] = models
        self.model_dir: str = model_dir
        self.save_output: bool = save_output
        self.output_dir: str = output_dir
        self.results: dict[str, dict] = {} # model_name -> {transcript: str, processing_time: float}

    def run_tests(self):

        for i, model_name in enumerate(self.models):
            print(f"\nTesting model: {model_name} ({i+1}/{len(self.models)})")
            transcriber = Transcriber(model_name=model_name, 
                                      model_dir=self.model_dir,
                                      device=self.device,
                                      compute_size=self.compute_size,
                                      batch_size=self.batch_size,
                                      diarize=self.diarize
                                      )
            start_time = time.time()
            transcript = tqdm(transcriber.transcribe(self.audio_path))
            end_time = time.time()
            processing_time = end_time - start_time

            print(f"Processing time for {model_name}: {processing_time:.2f} seconds")

            self.results[model_name] = {
                "transcript": transcript,
                "processing_time": processing_time
            }
        
        print(f"\nAll tests completed. Results:")
        print({model: {"processing_time": result["processing_time"]} for model, result in self.results.items()})

        if self.save_output:
            for model_name, result in self.results.items():
                with open(self.output_dir / f"transcript_{model_name}.txt", "w", encoding="utf-8") as f:
                    f.write(result["transcript"])
            
            with open(self.output_dir / "test_results.txt", "w", encoding="utf-8") as f:
                for model_name, result in self.results.items():
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Processing time: {result['processing_time']:.2f} seconds\n")
                    f.write("\n")
            

if __name__ == "__main__":
    tester = Tester(
        audio_path="input/transcoded.mp3",
        models=["whisperx_small", "whisperx_medium"],
        model_dir="model",
        output_dir="output",
        device="cpu",
        compute_size="int8",
        batch_size=16,
        diarize=False,
        save_output=True
    )
    tester.run_tests()