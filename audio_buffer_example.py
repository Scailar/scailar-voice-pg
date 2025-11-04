import datetime
import base64
# Simple Audio Frame Production and Consumption with Buffer

FRAME_SIZE = 3  # Number of audio frames to hold in the buffer
audio_buffer = []  # Initialize the audio buffer

# Simulated function to produce audio frames
def produce_audio_frames(frame_count):
    for i in range(frame_count):
        yield f"audio_frame_{i + 1}".encode()  # Generating dummy audio frames as bytes

# Simulated consumer function that sends frames when buffer is full
def process_buffer():
    if len(audio_buffer) >= FRAME_SIZE:
        # Create a single payload for all buffered frames
        combined_data = b''.join(audio_buffer)  # Combine all buffered frames into one byte string
        payload = {
            "kind": "AudioData",
            "audioData": {
                "timestamp": ts(),
                "participantRawID": "participant_id",  # Example participant ID
                "data": base64.b64encode(combined_data).decode(),
                "silent": False,
            },
        }
        # Simulate sending frames (printing for this example)
        print("Sending:", payload)
        # Clear the buffer after sending
        audio_buffer.clear()

# Simulated function to get the current timestamp
def ts() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

# Simulated producer-consumer loop
for frame in produce_audio_frames(10):  # Produce 10 audio frames
    audio_buffer.append(frame)  # Add frame to the buffer
    print("Buffer State:", audio_buffer)  # Show current buffer state
    process_buffer()  # Process buffer if it's full

# After all frames are produced, check if there are remaining frames
if audio_buffer:
    # Create a single payload for any remaining frames
    combined_data = b''.join(audio_buffer)
    payload = {
        "kind": "AudioData",
        "audioData": {
            "timestamp": ts(),
            "participantRawID": "participant_id",  # Example participant ID
            "data": base64.b64encode(combined_data).decode(),
            "silent": False,
        },
    }
    print("Sending remaining frames:", payload)