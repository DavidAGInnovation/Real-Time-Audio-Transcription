import pyaudio
import numpy as np
import whisper
import noisereduce as nr

# Initialize Whisper model
model = whisper.load_model("base")  # Using a larger model for better accuracy

# PyAudio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
BUFFER_SIZE = 40  # Number of chunks to accumulate (increase buffer for more context)
SILENCE_THRESHOLD = 0.002  # Less strict threshold to detect silence

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Start the stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

audio_buffer = []

def is_silent(audio_data, threshold=SILENCE_THRESHOLD):
    mean_amplitude = np.mean(np.abs(audio_data))
    # print(f"Mean amplitude: {mean_amplitude}")  # Debugging line
    return mean_amplitude < threshold

try:
    while True:
        try:
            # Read data from the stream
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Convert the data to numpy array and normalize
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer.append(audio_data)
            
            # Check if we have enough audio in buffer
            if len(audio_buffer) >= BUFFER_SIZE:
                # Concatenate the audio buffer into a single array
                audio_input = np.concatenate(audio_buffer, axis=0)
                
                # Clear the buffer for new data
                audio_buffer = []
                
                # Check if the audio is not silent
                if not is_silent(audio_input):
                    # print("Audio buffer is not silent, processing...")
                    # Perform noise reduction
                    audio_input = nr.reduce_noise(y=audio_input, sr=RATE)
                    # print("Noise reduction completed.")
                    
                    # Run the Whisper model
                    result = model.transcribe(audio_input, fp16=False)
                    
                    # Print the transcription
                    print(result['text'])
                else:
                    print("Silence detected, skipping transcription.")
                
        except OSError as e:
            if e.errno == -9981:
                print("Input overflowed. Skipping this chunk.")
                continue
            elif e.errno == -9988:
                print("Stream closed unexpectedly. Exiting.")
                break
            else:
                raise

except KeyboardInterrupt:
    print("Stopping...")

finally:
    # Stop and close the stream
    try:
        stream.stop_stream()
    except OSError:
        print("Error stopping stream.")
    
    try:
        stream.close()
    except OSError:
        print("Error closing stream.")
    
    audio.terminate()
    print("Terminated audio interface.")
