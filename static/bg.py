import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio
import numpy as np
import sounddevice as sd
import keyboard
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


os.environ["GOOGLE_API_KEY"] = "AIzaSyDXg5GinUuT4axifXMgNvIWkvtgs7NuepE"  

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

    waveform = np.concatenate(recording, axis=0)
    return np.squeeze(waveform)

def get_text_summary(text):
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set your Google API key as an environment variable 'GOOGLE_API_KEY'")

    try:
        llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.5)  # Adjust temperature as needed
    except Exception as e:
        print(f"Error creating LLM instance: {e}")
        return None

    # Define a prompt template
    prompt_template = PromptTemplate.from_template("Summarize the following text:\n{text}")

    # Create a chain using LangChain's method
    chain = (
        {"text": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Run the chain to generate the summary
    try:
        summary = chain.invoke(text)
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

# Function to transcribe audio and then summarize the transcription
def transcribe_and_summarize():
    option = input("Choose input method (1: File, 2: Record): ")

    if option == "1":
        audio_path = input("Enter the path to the audio file: ")
        waveform = process_audio_file(audio_path)
    elif option == "2":
        waveform = record_audio()
    else:
        print("Invalid option. Please choose 1 or 2.")
        return

    # Transcribe the audio
    result = pipe(waveform, return_timestamps=True, generate_kwargs={"task": "transcribe"})
    transcription = result["text"]
    print("Transcription:", transcription)

    # Summarize the transcription using Gemini API
    summary = get_text_summary(transcription)
    if summary:
        print("Summary:", summary)
    else:
        print("Could not generate a summary.")

if __name__ == "__main__":
    transcribe_and_summarize()