import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio
import numpy as np
import sounddevice as sd
import keyboard

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)


pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


def process_audio_file(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Whisper expects a single channel (mono) audio, ensure it's single-channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0).unsqueeze(0)

    waveform = waveform.numpy()
    waveform = np.squeeze(waveform)

    return waveform

def record_audio(sample_rate=16000):
    print("Recording... Press 'q' to stop.")
    recording = []

    def callback(indata):
        recording.append(indata.copy())

    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=callback)
    
    
    with stream:
        while True:
            if keyboard.is_pressed('q'):
                print("Recording stopped.")
                break

    # Concatenate all recorded chunks
    waveform = np.concatenate(recording, axis=0)
    return np.squeeze(waveform)

def transcribe_audio():
    option = input("Choose input method (1: File, 2: Record): ")

    if option == "1":
        audio_path = input("Enter the path to the audio file: ")
        waveform = process_audio_file(audio_path)
    elif option == "2":
        waveform = record_audio()
    else:
        print("Invalid option. Please choose 1 or 2.")
        return

    result = pipe(waveform, return_timestamps=True, generate_kwargs={"task": "transcribe"})
    print("Transcription:", result["text"])

if __name__ == "__main__":
    transcribe_audio()