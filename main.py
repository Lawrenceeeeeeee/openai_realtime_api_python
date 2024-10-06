# Updated code to fix audio playback issue by ensuring correct sampling rate and format.

import os
import websocket
import base64
import numpy as np
import wave
import json
import pyaudio
import ssl
import threading
import time
from queue import Queue

# websocket.enableTrace(True)

session_updated = False
playback_queue = Queue()  # Queue to hold audio data for sequential playback
playback_paused = threading.Event()  # Event to control playback pause

def float_to_16bit_pcm(float32_array):
    pcm16_array = np.clip(float32_array, -1, 1) * 32767
    return pcm16_array.astype(np.int16).tobytes()

def base64_encode_audio(float32_array):
    pcm16_data = float_to_16bit_pcm(float32_array)
    return base64.b64encode(pcm16_data).decode('utf-8')

def send_audio(ws, audio_chunk):
    float32_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
    base64_chunk = base64_encode_audio(float32_array)
    ws.send(
        json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64_chunk
        })
    )

def on_open(ws):
    print("WebSocket connection opened")
    # Send initial session update message
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": "你是一位语音助手，帮助用户完成各种任务。请你主要用中文回答用户的问题。",
            "voice": "alloy",
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 200
            },
            "temperature": 0.8,
        }
    }
    ws.send(json.dumps(session_update))

    # Start a new thread for recording audio after session update is confirmed
    audio_thread = threading.Thread(target=send_audio_stream, args=(ws,))
    audio_thread.start()

    # Start a new thread for playing audio from the queue
    playback_thread = threading.Thread(target=play_audio_from_queue)
    playback_thread.start()

def on_message(ws, message):
    global session_updated
    try:
        event = json.loads(message)
        print(event.get("type"))
        if event.get("type") == "session.updated":
            print("Session updated successfully.")
            session_updated = True
        elif event.get("type") == "response.audio.delta":
            audio_data_base64 = event.get("delta")
            if audio_data_base64:
                audio_data = base64.b64decode(audio_data_base64)
                # Add audio data to the playback queue
                playback_queue.put(audio_data)
        elif event.get("type") == "input_audio_buffer.speech_started":
            # Pause playback and clear the queue when user starts speaking
            playback_paused.set()
            with playback_queue.mutex:
                playback_queue.queue.clear()
        elif event.get("type") == "error":
            print(f"Error occurred: {event.get('error')}")
        elif event.get("type") == "response.audio_transcript.done":
            print(f"Assistant: {event.get('transcript')}")
        elif event.get("type") == "conversation.item.input_audio_transcription.completed":
            print(f"User: {event.get('transcript')}")
    except json.JSONDecodeError as e:
        print(f"Error decoding message: {e}")

def play_audio_from_queue():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=24000,
                    output=True)
    try:
        while True:
            playback_paused.wait()  # Wait if playback is paused
            audio_data = playback_queue.get()  # Block until there is audio data to play
            stream.write(audio_data)
    except Exception as e:
        print(f"Error while playing audio: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def on_error(ws, error):
    print(f"Error occurred: {error}")
    # # Retry connection on error
    # print("Attempting to reconnect in 5 seconds...")
    # time.sleep(5)
    # connect()

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")
    # # Retry connection on close
    # print("Attempting to reconnect in 5 seconds...")
    # time.sleep(5)
    # connect()

def send_audio_stream(ws):
    # Start recording audio from microphone
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=24000,
                    input=True,
                    frames_per_buffer=1024)
    print("Recording audio...")
    try:
        while True:
            audio_chunk = stream.read(1024, exception_on_overflow=False)
            send_audio(ws, audio_chunk)
    except Exception as e:
        print(f"Error while recording audio: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def connect():
    ws = websocket.WebSocketApp(
        url,
        header=[f"{key}: {value}" for key, value in headers.items()],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
headers = {
    "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY"),
    "OpenAI-Beta": "realtime=v1",
}

connect()