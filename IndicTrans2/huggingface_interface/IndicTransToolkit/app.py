from flask import Flask, render_template, request, jsonify
import threading
import os
import wave
import time
import math
import pyaudio
from pyannote.audio import Pipeline, Model, Inference
from scipy.spatial.distance import cdist
from pydub import AudioSegment
import pandas as pd
import numpy as np
import speech_recognition as sr
import csv
import openai
from dotenv import load_dotenv
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Constants
FORMAT = pyaudio.paInt24 #this is for virtual mic change it to paInt16 for physical mic
CHANNELS = 1
RATE = 48000    #this is for virtual mic change it to 16000 for physical mic
CHUNK_DURATION = 6 #this are the seconds we can change it from range 3-10 second 
buffer_rate=16000
CHUNK_DIR = "chunks"
SEGMENT_DIR = "segments"
ACCESS_TOKEN = "replace with hugging face token"
THRESHOLD = 0.8  # frequency to match the speaker simalarity using cosine


# State
recording = False
chunk_counter = 1
segment_counter = 1
recording_thread = None
last_speaker = None
transcript_lines = []
selected_language = "en-IN"
input_device_index=1
executor = ThreadPoolExecutor(max_workers=5) 
unknown_speaker_count = 1 
# Directories
os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(SEGMENT_DIR, exist_ok=True)

# Load models
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=ACCESS_TOKEN)
embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=ACCESS_TOKEN)
inference = Inference(embedding_model, window="whole")
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Speaker label storage
speaker_embeddings = []
segment_speakers = []
speaker_names = []



#indic
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None

# Transcription function
def extract_text_from_audio(audio_file_path, start_time, end_time):
    buffer=800
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source, duration=end_time+buffer, offset=start_time)
    try:
        return recognizer.recognize_google(audio, language=selected_language)
    except sr.UnknownValueError:
        return ""  
    except sr.RequestError as e:
        return f"(API error: {e})"


def diarize_and_segment(chunk_path, rttm_path):
    global segment_counter, speaker_embeddings, segment_speakers, last_speaker, transcript_lines

    print(f" Diarizing {chunk_path}...")
    diarization = pipeline(chunk_path)
    with open(rttm_path, "w") as f:
        diarization.write_rttm(f)

    print(f" RTTM saved to {rttm_path}")
    audio = AudioSegment.from_wav(chunk_path)

    df = pd.read_csv(rttm_path, sep=" ", header=None, comment=";", names=[
        "Type", "File ID", "Channel", "Start", "Duration",
        "NA1", "NA2", "Speaker", "NA3", "NA4"
    ])

    for _, row in df.iterrows():
        start = row["Start"]
        duration = row["Duration"]

        if duration < 0.5:
            continue

        buffer=800
        end = start + duration
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)+buffer 

        segment_audio = audio[start_ms:end_ms]
        segment_filename = f"segment_{segment_counter}.wav"
        segment_path = os.path.join(SEGMENT_DIR, segment_filename)
        segment_audio.export(segment_path, format="wav")
        print(f" Saved: {segment_filename} | Start={start:.3f}s Duration={duration:.3f}s")
        segment_counter += 1

        # Transcribe the segment
        transcript = extract_text_from_audio(segment_path, start_time=0, end_time=duration)
        if transcript.strip() == "":
            print(f" Skipping {segment_filename} — empty transcription")
            continue  # Skip embedding and labeling

        # Assign speaker label
        #pretrain
        nitik_sample_path = "D:/meeting poc/language_meet - Copy/IndicTrans2/huggingface_interface/IndicTransToolkit/voices/nitik.wav"  # Replace with actual path
        nitik_embedding = inference(nitik_sample_path).reshape(1, -1)
        speaker_embeddings.append(nitik_embedding)
        speaker_names.append("Nitik")

        nitik_sample_path2 = "D:/meeting poc/language_meet - Copy/IndicTrans2/huggingface_interface/IndicTransToolkit/voices/nitik2.wav"  # Replace with actual path
        nitik_embedding2 = inference(nitik_sample_path2).reshape(1, -1)
        speaker_embeddings.append(nitik_embedding2)
        speaker_names.append("Nitik")

        nitik_sample_path3 = "D:/meeting poc/language_meet - Copy/IndicTrans2/huggingface_interface/IndicTransToolkit/voices/nitik3.wav"  # Replace with actual path
        nitik_embedding3 = inference(nitik_sample_path3).reshape(1, -1)
        speaker_embeddings.append(nitik_embedding3)
        speaker_names.append("Nitik")

        indu_sample_path="D:/meeting poc/language_meet - Copy/IndicTrans2/huggingface_interface/IndicTransToolkit/voices/indu.wav"
        indu_embedding = inference(indu_sample_path).reshape(1, -1)
        speaker_embeddings.append(indu_embedding)
        speaker_names.append("Indu")

        indu_sample_path2="D:/meeting poc/language_meet - Copy/IndicTrans2/huggingface_interface/IndicTransToolkit/voices/indu2.wav"
        indu_embedding2 = inference(indu_sample_path2).reshape(1, -1)
        speaker_embeddings.append(indu_embedding2)
        speaker_names.append("Indu")

        indu_sample_path3="D:/meeting poc/language_meet - Copy/IndicTrans2/huggingface_interface/IndicTransToolkit/voices/indu3.wav"
        indu_embedding3 = inference(indu_sample_path3).reshape(1, -1)
        speaker_embeddings.append(indu_embedding3)
        speaker_names.append("Indu")

        ramya_sample_path="D:/meeting poc/language_meet - Copy/IndicTrans2/huggingface_interface/IndicTransToolkit/voices/ramya.wav"
        ramya_embedding = inference(ramya_sample_path).reshape(1, -1)
        speaker_embeddings.append(ramya_embedding)
        speaker_names.append("Ramya")

        ramya_sample_path2="D:/meeting poc/language_meet - Copy/IndicTrans2/huggingface_interface/IndicTransToolkit/voices/ramya2.wav"
        ramya_embedding2 = inference(ramya_sample_path2).reshape(1, -1)
        speaker_embeddings.append(ramya_embedding2)
        speaker_names.append("Ramya")

        ramya_sample_path3="D:/meeting poc/language_meet - Copy/IndicTrans2/huggingface_interface/IndicTransToolkit/voices/ramya3.wav"
        ramya_embedding3 = inference(ramya_sample_path3).reshape(1, -1)
        speaker_embeddings.append(ramya_embedding3)
        speaker_names.append("Ramya")

        vidit_sample_path="D:/meeting poc/language_meet - Copy/IndicTrans2/huggingface_interface/IndicTransToolkit/voices/vidit.wav"
        vidit_embedding = inference(vidit_sample_path).reshape(1, -1)
        speaker_embeddings.append(vidit_embedding)
        speaker_names.append("Vidit")

        vidit_sample_path2="D:/meeting poc/language_meet - Copy/IndicTrans2/huggingface_interface/IndicTransToolkit/voices/vidit2.wav"
        vidit_embedding2 = inference(vidit_sample_path2).reshape(1, -1)
        speaker_embeddings.append(vidit_embedding2)
        speaker_names.append("Vidit")

        emb = inference(segment_path).reshape(1, -1)
        print(f'current={emb}')
        if not speaker_embeddings:
            speaker_embeddings.append(emb)
            speaker_label = "Speaker_1"
            print(f" {segment_filename} → {speaker_label} (first speaker)")
        else:
            distances = [cdist(emb, known_emb, metric="cosine")[0, 0] for known_emb in speaker_embeddings]
            min_dist = min(distances)

            if min_dist <= THRESHOLD:
                speaker_idx = distances.index(min_dist)
                speaker_label = speaker_names[speaker_idx]
            else:
                new_label = f"unknown_speaker_{unknown_speaker_count}"
                unknown_speaker_count += 1  # Increment for next unknown speaker
                speaker_embeddings.append(emb)
                speaker_names.append(new_label)
                speaker_label = new_label

            print(f"{segment_filename} → {speaker_label} (min_dist={min_dist:.4f})")
            segment_speakers.append((segment_filename, speaker_label))
            print(f"{speaker_label}: {transcript}")

        # Merge if same speaker as previous
        if speaker_label == last_speaker and transcript_lines:
            transcript_lines[-1] = transcript_lines[-1].strip() + f" {transcript}"
        else:
            transcript_lines.append(f"{speaker_label}: {transcript}")
        last_speaker = speaker_label

        # Write the updated transcript to file
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines))

def record_chunks():
    global recording, chunk_counter,input_device_index,buffer_rate

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=buffer_rate,input_device_index=input_device_index)

    try:
        while recording:
            frames = []
            for _ in range(math.ceil(RATE / buffer_rate * CHUNK_DURATION)):
                if not recording:
                    break
                data = stream.read(buffer_rate)
                frames.append(data)

            if frames:
                chunk_path = os.path.join(CHUNK_DIR, f"chunk_{chunk_counter}.wav")
                wf = wave.open(chunk_path, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

                print(f"\n📦 Saved chunk: {chunk_path}")
                rttm_path = chunk_path.replace(".wav", ".rttm")
                executor.submit(diarize_and_segment, chunk_path, rttm_path)

                chunk_counter += 1
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def getTranslation(content):
    def initialize_model_and_tokenizer(ckpt_dir, quantization):
        if quantization == "4-bit":
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8-bit":
            qconfig = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        else:
            qconfig = None

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=qconfig,
        )

        if qconfig == None:
            model = model.to(DEVICE)
            if DEVICE == "cuda":
                model.half()

        model.eval()

        return tokenizer, model


    def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
        translations = []
        for i in range(0, len(input_sentences), BATCH_SIZE):
            batch = input_sentences[i : i + BATCH_SIZE]

            # Preprocess the batch and extract entity mappings
            batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

            # Tokenize the batch and generate input encodings
            inputs = tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(DEVICE)

            # Generate translations using the model
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )

            # Decode the generated tokens into text

            with tokenizer.as_target_tokenizer():
                generated_tokens = tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            # Postprocess the translations, including entity replacement
            translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

            del inputs
            torch.cuda.empty_cache()

        return translations

    indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"  # ai4bharat/indictrans2-indic-en-dist-200M
    indic_en_tokenizer, indic_en_model = initialize_model_and_tokenizer(indic_en_ckpt_dir, quantization)

    ip = IndicProcessor(inference=True)
    if selected_language=="hi-IN":
        src_lang, tgt_lang = "hin_Deva", "eng_Latn"
    elif selected_language=="ta-IN":
        src_lang, tgt_lang = "tam_Taml", "eng_Latn"
    elif selected_language=="te-IN":
        src_lang, tgt_lang = "tel_Telu", "eng_Latn"
    elif selected_language=="bn-IN":
        src_lang, tgt_lang = "ben_Beng", "eng_Latn"
    elif selected_language=="gu-IN":
        src_lang, tgt_lang = "guj_Gujr", "eng_Latn"
    elif selected_language=="kn-IN":
        src_lang, tgt_lang = "kan_Knda", "eng_Latn"
    elif selected_language=="ml-IN":
        src_lang, tgt_lang = "mal_Mlym", "eng_Latn"
    elif selected_language=="mr-IN":
        src_lang, tgt_lang = "mar_Deva", "eng_Latn"
    elif selected_language=="pa-IN":
        src_lang, tgt_lang = "pan_Guru", "eng_Latn"
    elif selected_language=="ur-IN":
        src_lang, tgt_lang = "urd_Arab", "eng_Latn"
    en_translations = batch_translate(content, src_lang, tgt_lang, indic_en_model, indic_en_tokenizer, ip)
    

    print(f"\n{src_lang} - {tgt_lang}")
    for input_sentence, translation in zip(content, en_translations):
        print(f"{src_lang}: {input_sentence}")
        print(f"{tgt_lang}: {translation}")
        
    
    return en_translations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global recording, recording_thread, chunk_counter, segment_counter
    global speaker_embeddings, segment_speakers, selected_language, input_device_index  # <-- Add input_device_index here

    selected_language = request.args.get("lang", "en-IN")
    input_device_index = int(request.args.get("input_device_index", 3))  # <-- Get value from frontend

    if not recording:
        recording = True
        chunk_counter = 1
        segment_counter = 1
        speaker_embeddings = []
        segment_speakers = []

        recording_thread = threading.Thread(target=record_chunks)
        recording_thread.start()
        return jsonify({'status': f'🎙️Recording started using device {input_device_index}'})

    return jsonify({'status': '✅ Already recording'})


@app.route('/stop', methods=['POST'])
def stop():
    global recording, recording_thread, chunk_counter, executor

    recording = False

    if recording_thread is not None:
        recording_thread.join()
        print("🛑 Recording thread stopped.")

    # Handle final partial chunk if exists
    last_chunk_index = chunk_counter
    last_chunk_path = os.path.join(CHUNK_DIR, f"chunk_{last_chunk_index}.wav")
    rttm_path = last_chunk_path.replace(".wav", ".rttm")

    # Only process the last chunk if it hasn't already
    if os.path.exists(last_chunk_path) and not os.path.exists(rttm_path):
        print(f"🧩 Processing final chunk: chunk_{last_chunk_index}.wav")
        future = executor.submit(diarize_and_segment, last_chunk_path, rttm_path)
        future.result()  # Wait for it to finish

    # Shutdown executor after all diarization tasks
    executor.shutdown(wait=True)

    # Save speaker labels from processed segments
    if segment_speakers:
        df = pd.DataFrame(segment_speakers, columns=["segment", "speaker"])
        df.to_csv("detected_speakers.csv", index=False)
        print("✅ Speaker labels saved to detected_speakers.csv")

    # Write final transcript if not already
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(transcript_lines))
    print("✅ Transcript finalized.")

    return jsonify({
        'status': '🛑 Recording stopped'
    })


import os

@app.route('/clear', methods=['POST'])
def clear():
    for folder in [CHUNK_DIR, SEGMENT_DIR]:
        if os.path.exists(folder):
            print(f"Cleaning up {folder}: {os.listdir(folder)}")
            for f in os.listdir(folder):
                try:
                    os.remove(os.path.join(folder, f))
                except Exception as e:
                    print(f"Error deleting file {f}: {e}")
        else:
            print(f"Folder {folder} does not exist.")
    
    # Remove speaker CSV
    if os.path.exists("detected_speakers.csv"):
        os.remove("detected_speakers.csv")
        print(" detected_speakers.csv removed.")
    
    # Remove transcript file
    if os.path.exists("transcript.txt"):
        os.remove("transcript.txt")
        print(" transcript.txt removed.")

    return jsonify({'status': '✅All chunks, segments, transcript, and CSV cleared'})



@app.route('/get_transcript')
def get_transcript():
    transcript_path = "D:/meeting poc/language_meet - Copy/IndicTrans2/huggingface_interface/IndicTransToolkit/transcript.txt"
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Filter out lines where the content is None or empty after colon
        cleaned_lines = [line.strip() for line in lines if ":" in line and line.strip().split(":", 1)[1].strip().lower() not in ["", "none"]]
        return jsonify({"transcript": "\n\n".join(cleaned_lines)})
    return jsonify({"transcript": ""})


@app.route('/get_summary', methods=['GET'])
def get_summary():
    global summary_ready
    transcript_path = "D:/meeting poc/language_meet - Copy/IndicTrans2/huggingface_interface/IndicTransToolkit/transcript.txt"
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    else:
        full_text = ""

    if full_text.strip():
        prompt = f"""
        You are a helpful assistant. Please read the following meeting transcript and return the following:

        1. A complete summary of the conversation 
        2. Key discussion points (as bullet points)
        3. Action items (as bullet points)

        Transcript:
        {full_text}

        Format your response as:
        Summary: ...
        Key Points:
        - ...
        Actions:
        - ...
        """
        try:
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            content = response['choices'][0]['message']['content']
            summary_part = content.split("Key Points:")[0].replace("Summary:", "").strip()
            keypoints_part = content.split("Key Points:")[1].split("Actions:")[0].strip()
            actions_part = content.split("Actions:")[1].strip()

            app.config["SUMMARY"] = {
                "summary": summary_part,
                "key_points": keypoints_part,
                "actions": actions_part
            }
            summary_ready = True
            print("✅ Summary generation complete.")
        except Exception as e:
            print(f"❌ Error during summary generation: {e}")
            app.config["SUMMARY"] = {"summary": "", "key_points": "", "actions": ""}
            summary_ready = True
    else:
        print("⚠️ Empty transcript. No summary generated.")
        app.config["SUMMARY"] = {"summary": "", "key_points": "", "actions": ""}
        summary_ready = True
    
        

    return jsonify(app.config.get("SUMMARY", {
        "summary": "",
        "key_points": "",
        "actions": ""
    }))

@app.route('/get-translation')
def get_translation():
    transcript_path = "transcript.txt"
    if not os.path.exists(transcript_path):
        return jsonify({'translation': "Transcript not found."})

    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()
        content=[content]
    translation = getTranslation(content)
    return jsonify({'translation': translation})


if __name__ == '__main__':
    app.run(debug=True)
