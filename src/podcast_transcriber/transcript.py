"""Transcript helpers."""

from pathlib import Path
from pyannote import audio
import whisperx
import gc
from whisperx.diarize import DiarizationPipeline
import torch
from tqdm import tqdm
import os
import dotenv
import whisper
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from qwen_asr import Qwen3ASRModel
from opencc import OpenCC


dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") # get from https://huggingface.co/settings/tokens

SCRIPT_DIR = Path(__file__).parent

AVAILABLE_MODELS = ["whisperx_small", 
                    "whisperx_medium", 
                    "whisper_small", 
                    "whisper_medium",
                    "sensevoice_small",
                    "sensevoice_large",
                    "qwen3_asr",]

NUM_CORES = os.cpu_count() or 1
os.environ["OMP_NUM_THREADS"] = str(NUM_CORES) # num of cores used
os.environ["MKL_NUM_THREADS"] = str(NUM_CORES) # num of cores used


class Transcriber:

    def __init__(self, model_name: str = "whisperx_small", 
                 model_dir: str = None, 
                 device: str = "cpu", 
                 compute_type: str = "int8",
                 batch_size: int = 16,
                 diarize: bool = False,
                 align: bool = False,
                 chi_sim: bool = False,         # True for simplified Chinese, False for traditional Chinese.
                 ):
        
        self.model_name: str = model_name
        if self.model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Model '{self.model_name}' not supported. Available models: {AVAILABLE_MODELS}")
        
        self.model_dir: str = model_dir
        self.device: str = device
        self.compute_type: str = compute_type
        self.batch_size: int = batch_size
        self.diarize: bool = diarize
        self.align: bool = align

        self.chi_sim: bool = chi_sim
        self.cc = OpenCC('t2s') if chi_sim else OpenCC('s2t') # converter for Chinese text

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
        if self.align:
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

        # 4. Convert traditional/simplified Chinese if needed
        for segment in result["segments"]:
            segment["text"] = segment["text"].strip()
            if self.chi_sim:
                segment["text"] = self.cc.convert(segment["text"])
            else:
                segment["text"] = self.cc.convert(segment["text"])

        self.transcript = result

        return self.transcript

    def _transcribe_whisper(self, model_size: str, audio_path: str | Path) -> dict:
        """Transcribe the given audio file and return the transcript text."""
        
        # 1. Transcribe with whisper
        self.model: whisper.Whisper = whisper.load_model(model_size, device=self.device)

        print("\n" + "="*50 + "\nTranscribing audio with Whisper...")
        result = self.model.transcribe(str(audio_path))
        print(result["segments"])

        # 2. Align whisper output
        if self.align:
            print("\n" + "="*50 + "\nAligning whisper output...")
            audio = whisperx.load_audio(str(audio_path))
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            cleaned_result = [{"text": segment["text"], "start": segment["start"], "end": segment["end"]} 
                              for segment in result["segments"] if segment["text"].strip() != ""] # remove empty segments and keep only text, start and end
            result = whisperx.align(cleaned_result, model_a, metadata, audio, self.device, return_char_alignments=False)
            print(result["segments"]) # after alignment

        # 4. Convert traditional/simplified Chinese if needed
        for segment in result["segments"]:
            segment["text"] = segment["text"].strip()
            if self.chi_sim:
                segment["text"] = self.cc.convert(segment["text"])
            else:
                segment["text"] = self.cc.convert(segment["text"])

        self.transcript = result

        return self.transcript
    
    def _transcribe_sensevoice(self, model_size: str, audio_path: str | Path) -> dict:
        """Transcribe the given audio file and return the transcript text."""

        model_dir = "iic/SenseVoiceSmall" if model_size == "small" else "iic/SenseVoiceLarge"
        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            remote_code="./model.py",    
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=self.device,
        )

        res = model.generate(
            input=str(audio_path),
            cache={},
            language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )

        # post-process the raw transcription results
        self.transcript = {"text": rich_transcription_postprocess(res[0]["text"])}
    
        if self.chi_sim:
            self.transcript["text"] = self.cc.convert(self.transcript["text"])
        else:
            self.transcript["text"] = self.cc.convert(self.transcript["text"])

        print(self.transcript)
        return self.transcript

    def _transcribe_qwen3_asr(self, audio_path: str | Path) -> dict:
        """Transcribe the given audio file and return the transcript text."""

        model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            dtype=torch.bfloat16,
            device_map="auto",
            # attn_implementation="flash_attention_2",
            max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
            max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map="auto",
                # attn_implementation="flash_attention_2",
            ),
        )

        results = model.transcribe(
            audio=[str(audio_path)],
            language=None, # can also be set to None for automatic language detection
            return_time_stamps=True,
        )

        print(results)

        self.transcript = results[0] # for simplicity, we only handle single audio input here
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
        
        # 2. Whisper model
        if self.model_name.startswith("whisper"):
            model_size = self.model_name.split("_")[1] # e.g. "small" or "medium"
            return self._transcribe_whisper(model_size=model_size, audio_path=audio_path)

        # 3. SenseVoice ASR model
        if self.model_name.startswith("sensevoice"):
            model_size = self.model_name.split("_")[1] # e.g. "small" or "medium"
            return self._transcribe_sensevoice(model_size=model_size, audio_path=audio_path)

        if self.model_name.startswith("qwen3_asr"):
            return self._transcribe_qwen3_asr(audio_path=audio_path)


    def export_to_txt(self, output_path: str | Path) -> str:
        """Export the transcript text to a .txt file and return the file path."""
        
        print("\nExporting transcript to text file...")
        if self.transcript is None:
            raise ValueError("No transcript available. Please transcribe an audio file first.") 
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            print("Writing transcript to file...")

            if self.transcript.get("segments") is None:
                # if no segments, just write the full text
                f.write(self.transcript.get("text", ""))
                print("No segments found in transcript. Wrote full text to file.")
                return str(output_path)

            for segment in tqdm(self.transcript["segments"], desc="Exporting segments"):
                start_time = segment["start"]
                end_time = segment["end"]
                speaker_id = segment.get("speaker", "Unknown")
                text = segment["text"].strip()
                f.write(f"[{start_time:.2f} - {end_time:.2f}] Speaker {speaker_id}: {text}\n")

        return str(output_path)


if __name__ == "__main__":

    # AVAILABLE_MODELS = ["whisperx_small",
    #                     "whisperx_medium", 
    #                     "whisper_small", 
    #                     "whisper_medium",
    #                     "sensevoice_small",
    #                     "sensevoice_large",
    #                     "qwen3_asr",]

    model_name = "sensevoice_small"
    align = False
    
    transcriber = Transcriber(model_name=model_name, model_dir=SCRIPT_DIR/"../../model", align=align)
    transcriber.transcribe(audio_path=SCRIPT_DIR / "../../input/debug_canto.mp3")
    sample_transcript = "This is a sample transcript for debugging."
    output_file = SCRIPT_DIR / "../../output/debug_canto.txt"
    saved_path = transcriber.export_to_txt(output_file)
    