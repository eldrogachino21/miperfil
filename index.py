# Real-Time Spanish to English Translation System Implementation Guide

## Prerequisites


## Step 1: Import Required Libraries

import speech_recognition as sr
from transformers import MarianMTModel, MarianTokenizer
import torch
from googletrans import Translator
from gtts import gTTS
import pygame
import sounddevice as sd
import numpy as np
import queue
import threading
import time

## Step 2: Initialize Components

def initialize_components():
    # Initialize speech recognizer
    recognizer = sr.Recognizer()
    
    # Initialize translator
    translator = Translator()
    
    # Initialize MT model
    model_name = "Helsinki-NLP/opus-mt-es-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Initialize audio queue
    audio_queue = queue.Queue()
    
    return recognizer, translator, model, tokenizer, audio_queue

## Step 3: Audio Capture Function

def capture_audio(audio_queue, sample_rate=16000):
    def audio_callback(indata, frames, time, status):
        audio_queue.put(bytes(indata))
    
    stream = sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=sample_rate,
        dtype=np.int16
    )
    
    return stream

def process_audio_chunk(audio_data, recognizer):
    audio = sr.AudioData(audio_data, 16000, 2)
    try:
        text = recognizer.recognize_google(audio, language='es-ES')
        return text
    except sr.UnknownValueError:
        return None


## Step 4: Translation Function

def translate_text(text, model, tokenizer, translator):
    try:
        # First attempt with MarianMT
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except:
        # Fallback to googletrans
        translated = translator.translate(text, src='es', dest='en').text
    
    return translated


## Step 5: Text-to-Speech Function

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    
    pygame.mixer.init()
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

## Step 6: Main Real-Time Processing Loop

def main():
    # Initialize components
    recognizer, translator, model, tokenizer, audio_queue = initialize_components()
    
    # Start audio stream
    stream = capture_audio(audio_queue)
    stream.start()
    
    print("Starting real-time translation...")
    
    try:
        while True:
            # Process audio in chunks
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                
                # Speech to text
                text = process_audio_chunk(audio_data, recognizer)
                
                if text:
                    print(f"Recognized: {text}")
                    
                    # Translate
                    translated = translate_text(text, model, tokenizer, translator)
                    print(f"Translated: {translated}")
                    
                    # Text to speech
                    text_to_speech(translated)
            
            time.sleep(0.1)  # Prevent CPU overload
            
    except KeyboardInterrupt:
        print("Stopping translation system...")
        stream.stop()


## Step 7: Performance Optimization Tips

# 1. Buffer Management
BUFFER_SIZE = 8192  # Adjust based on your needs

# 2. Implement threading for parallel processing
def threaded_processing():
    translation_thread = threading.Thread(target=translate_text)
    translation_thread.start()

# 3. Implement caching for frequent translations
translation_cache = {}
def cached_translation(text, model, tokenizer, translator):
    if text in translation_cache:
        return translation_cache[text]
    result = translate_text(text, model, tokenizer, translator)
    translation_cache[text] = result
    return result


## Usage Example

if __name__ == "__main__":
    main()

