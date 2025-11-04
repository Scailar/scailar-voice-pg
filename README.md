# README for Scailar Playground

This is a playground for demonstrating voice streaming capabilities for both inbound and outbound calls. Happy browsing! 

## Overview

`outbound_live_main.py` is a FastAPI application that integrates Azure Communication Services with Google Gemini for real-time audio processing. The system is designed to handle telephone calls while processing audio data, enabling a seamless interaction in healthcare settings, particularly for conducting health checks in palliative care.

## Features

- **Real-time Audio Processing**: Utilizes Google Gemini to process audio in real-time during telephone calls.
- **WebSocket Communication**: Establishes a WebSocket connection for bidirectional audio streaming.
- **Call Management**: Initiates calls and handles incoming events from Azure Communication Services.
- **Audio Resampling**: Converts audio between different sample rates for compatibility.

## Prerequisites

Before running this application, make sure you have:

- Python 3.7 or higher
- Required Python packages installed (FastAPI, Azure SDK, Google Gemini SDK, etc.)
- Environment variables set for:
  - `ACS_CONNECTION_STRING`: Your Azure Communication Services connection string.
  - `ACS_PUBLIC_BASE`: The base URL for public access (e.g., ngrok).
  - `GEMINI_API_KEY`: Your API key for Google Gemini.

## Running the Application

To run the application, execute the following command in your terminal:

```bash
python outbound_live_main.py
```

The FastAPI application will start and listen for incoming requests on `http://0.0.0.0:5503`.

## API Endpoints

### Dial

- **Endpoint**: `POST /dial`
- **Description**: Initiates a call to a specified phone number.
- **Request Body**:
  ```json
  {
    "to": "<target_phone_number>"
  }
  ```

### Events

- **Endpoint**: `POST /events`
- **Description**: Receives events related to the call, such as status updates.

### Media

- **Endpoint**: `WS /media`
- **Description**: WebSocket for bidirectional audio streaming. 

## Achieving Low Latency

Low latency in the `outbound_live_main.py` application is achieved through several design choices and implementations across multiple functions. Here's an overview of how the system minimizes delays in audio processing and transmission:

### 1. Bidirectional WebSocket Communication
The use of WebSockets for real-time audio streaming is a key factor in achieving low latency. Unlike traditional HTTP requests, WebSockets provide a persistent, full-duplex communication channel, allowing audio data to be sent and received concurrently without the overhead of establishing a new connection for each transmission.

- **Function**: `@app.websocket("/media")`

### 2. Asynchronous Tasks
The application employs Python's `asyncio` library to allow for asynchronous processing of tasks, significantly reducing potential bottlenecks during I/O operations.

- **Function**: `async def gemini_loop()`
- **Function**: `async def play_to_acs()`
- **Function**: `async def pull_from_acs()`

### 3. Efficient Buffering and Frame Processing
The application utilizes buffering techniques to handle audio data. By collecting multiple frames before sending them, the application optimizes bandwidth usage and maintains a smooth audio stream.

- **Function**: `async def play_to_acs()`

### 4. Real-time Audio Processing
The system processes audio in real-time, enabling quick reactions to incoming data and minimizing delays in the interaction.

- **Function**: `async def feed()` within `async def gemini_loop()`

### 5. Continuous Audio Retrieval and Processing
The application maintains a continuous loop for retrieving and processing audio data, ensuring readiness for new audio without idle delays.

- **Function**: `while True` in `async def pull_from_acs()`
