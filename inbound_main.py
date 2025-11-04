# main.py ─── all-Google speech, ACS telephony
import base64
import datetime
import json
import os, uuid, time, threading, queue, urllib.parse, asyncio, uvicorn, requests
from fastapi import FastAPI, Request, Response, WebSocket, Form
from google.cloud import texttospeech, speech_v1p1beta1 as speech
from azure.communication.callautomation import (
    CallAutomationClient,
    MediaStreamingOptions, MediaStreamingTransportType,
    MediaStreamingContentType, MediaStreamingAudioChannelType,
    AudioFormat,
    FileSource
)
from pydantic import BaseModel
import asyncio
from google import genai
from google.genai import types

# ───────────────────────────── env vars ────────────────────────────────────────
ACS_CONN_STR              = os.getenv("ACS_CONNECTION_STRING")        # full connection string
PUBLIC_BASE               = os.getenv("ACS_PUBLIC_BASE")              # https://your-app.azurewebsites.net
PHONE_NUMBER              = os.getenv("ACS_PHONE_NUMBER")             # +49…
AZ_OPENAI_ENDPOINT        = os.getenv("AZURE_OPENAI_SERVICE_ENDPOINT")
AZ_OPENAI_KEY             = os.getenv("AZURE_OPENAI_SERVICE_KEY")
AZ_OPENAI_DEPLOYMENT      = os.getenv("AZURE_OPENAI_DEPLOYMENT_MODEL_NAME")
AZ_OPENAI_API_VERSION     = os.getenv("AZURE_OPENAI_DEPLOYMENT_VERSION")  # e.g. 2024-02-15

# ───────────────────────────── clients ─────────────────────────────────────────
call_client  = CallAutomationClient.from_connection_string(ACS_CONN_STR)
gcp_tts      = texttospeech.TextToSpeechClient()

app          = FastAPI()
audio_cache  = {}                       # clip_id ➜ (bytes, timestamp)
call_state   = {}                       # call_id ➜ {"is_playing":bool}

# ─── tiny in-memory cache expiry ───────────────────────────────────────────────
def _purge_cache():
    while True:
        now = time.time()
        for k, (_, ts) in list(audio_cache.items()):
            if now - ts > 120:
                audio_cache.pop(k, None)
        time.sleep(30)
threading.Thread(target=_purge_cache, daemon=True).start()

# ───────────────────────────── helpers ─────────────────────────────────────────
def gpt_answer(history: list[dict]) -> str:
    """
    history = running list of {"role": ..., "content": ...}
    (last item must be the latest user message)
    """
    url     = f"{AZ_OPENAI_ENDPOINT}/openai/deployments/{AZ_OPENAI_DEPLOYMENT}/chat/completions"
    headers = {"api-key": AZ_OPENAI_KEY}
    payload = {
        "messages": history,
        "temperature": 0.7,
        "max_tokens": 200
    }
    params  = {"api-version": AZ_OPENAI_API_VERSION}

    r = requests.post(url, headers=headers, params=params,
                      json=payload, timeout=15)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]



client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chat = client.chats.create(model="gemini-2.0-flash", config=types.GenerateContentConfig(
        system_instruction="Du bist ein hilfreicher Call Center Agent. Antworte kurz und klar."),)
welcome_text = "Herzlich Willkommen in unserer Augenarztpraxis ProWischn, in Idstein! Wir möchten gerne ein paar Informationen erfassen, um Ihnen bestmöglich zu helfen. Bitte nennen Sie mir zunächst Ihr Anliegen. Worum geht es!"

def gemini_answer(message) -> str:
    """
    history = [{'role':'user'|'assistant', 'content': str}, … ]
    Returns Gemini-Flash reply text.
    """
    resp = chat.send_message(message)
    return resp.text.strip()


def synthesize(text:str) -> bytes:
    """German HD voice → MP3 in RAM → return HTTPS URL ACS can play."""
    voice = texttospeech.VoiceSelectionParams(
        language_code="de-DE",
        name="de-DE-Chirp3-HD-Achernar",
    )
    audio_cfg = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
    )
    resp = gcp_tts.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=voice,
        audio_config=audio_cfg,
    )
    return resp.audio_content
    clip_id    = str(uuid.uuid4())
    audio_cache[clip_id] = (resp.audio_content, time.time())
    return f"{PUBLIC_BASE}/audio/{clip_id}"

FRAME_MS   = 100
FRAME_LEN  = 16000 * 2 * FRAME_MS // 1000   # 3200 bytes
SILENCE_FRAME = b"\x00" * FRAME_LEN 
async def send_pcm_over_ws(
        ws,
        pcm: bytes,
        audio_q,
        participant_raw_id: str,
        cancel_event: asyncio.Event         # 👈 new
) -> None:

    ts_base = datetime.datetime.utcnow()

    for i in range(0, len(pcm), FRAME_LEN):
        # ↳ bail out immediately if barge-in happened
        if cancel_event.is_set():
            break

        slice = pcm[i:i + FRAME_LEN]
        timestamp = (
            ts_base + datetime.timedelta(milliseconds=i // 32)
        ).isoformat(timespec="milliseconds") + "Z"

        payload = {
            "Kind": "AudioData",
            "AudioData": {
                "Timestamp": timestamp,
                "ParticipantRawID": participant_raw_id,
                "Data": base64.b64encode(slice).decode(),
                "Silent": False,
            },
            "StopAudio": None,
        }

        await ws.send_text(json.dumps(payload))
        audio_q.put(SILENCE_FRAME)          # keep STT alive
        await asyncio.sleep(FRAME_MS / 1000)

    # tell ACS explicitly to stop any buffered audio (optional but tidy)
    await stop_playback(ws)


async def stop_playback(ws):
    data = {
            "Kind": "StopAudio",
            "AudioData": None,
            "StopAudio": {}
        }
    await ws.send_text(json.dumps(data))

# ───────────────────────────── models ──────────────────────────────────────────
class IncomingCall(BaseModel):
    incomingCallContext: str
    dataVersion: str | None = None
    # … other EventGrid fields omitted for brevity

# ───────────────────────────── endpoints ───────────────────────────────────────
@app.post("/incoming-call/")
async def incoming_call(request: Request):
    """
    Handles the Event Grid handshake AND real IncomingCall events.
    """
    body = await request.json()

    # 1) subscription validation
    if isinstance(body, list) and body and \
       body[0]["eventType"] == "Microsoft.EventGrid.SubscriptionValidationEvent":
        return {"validationResponse": body[0]["data"]["validationCode"]}

    ev  = body[0]                         # ACS bundles single events in a list
    if ev["data"]["to"]["phoneNumber"]["value"] != PHONE_NUMBER:
        return {"status": "ignored"}      # unknown number (multi-tenant)

    incall = ev["data"]
    caller_id = ev["data"]["to"]["phoneNumber"]["value"][1:]

    # 2) answer the call with media streaming
    call = call_client.answer_call(
        incoming_call_context = incall["incomingCallContext"],
        callback_url          = f"{PUBLIC_BASE}/acs-events/?call_id={caller_id}",
        media_streaming = MediaStreamingOptions(
            transport_url        = f'ws://replace/acs-media/?call_id={caller_id}',
            transport_type       = MediaStreamingTransportType.WEBSOCKET,
            content_type         = MediaStreamingContentType.AUDIO,
            audio_channel_type   = MediaStreamingAudioChannelType.UNMIXED,
            audio_format         = AudioFormat.PCM16_K_MONO,
            start_media_streaming= True,
            enable_bidirectional=True
        )
    )
    call_state[caller_id] = {
        "is_playing": False,
        "greeted":     False,
        "history": [],
    }

    return {"status": "answered"}

@app.post("/acs-events/")
async def acs_events(call_id, req: Request):
    """
    Only used to catch CallConnected so we can greet immediately.
    """
    body = await req.json()
    for ev in body:
        if ev["type"] == "Microsoft.Communication.CallConnected":
            # greet_url = synthesize("Hallo, wie kann ich Ihnen helfen?")
            # call_client.get_call_connection(call_id)\
            #            .play_audio_from_url(greet_url, loop=1)
            call_state[call_id]["is_playing"] = False
    return {"ok": True}


@app.websocket("/acs-media/")
async def acs_media(call_id, ws: WebSocket):
    """
    Receives 16-kHz PCM from ACS ➜ streams to GCP STT ➜ drives GPT & TTS.
    """
    await ws.accept()
    participant_raw_id = None
    qparams = urllib.parse.parse_qs(ws.url.query)
    call_id = qparams.get("call_id", [""])[0]

    if not call_id:
        await ws.close()
        return

    # ── function to launch one recogniser turn ───────────────────────────────
    def launch_recogniser():
        audio_q      = queue.Queue()
        transcript_q = queue.Queue()

        def stt_stream():
            client = speech.SpeechClient()
            recog_cfg = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="de-DE",
                enable_automatic_punctuation=True,
            )
            stream_cfg = speech.StreamingRecognitionConfig(
                config=recog_cfg,
                interim_results=True,
                single_utterance=False,          # Google ends on silence
            )

            def req_iter():
                for chunk in iter(audio_q.get, None):          # blocks
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)

            for resp in client.streaming_recognize(stream_cfg, req_iter()):
                for res in resp.results:
                    if res.alternatives:
                        transcript_q.put(
                            (res.is_final, res.alternatives[0].transcript.strip())
                        )
            # stream closed – poison pill any waiter then exit
            # transcript_q.put((True, ""))

        threading.Thread(target=stt_stream, daemon=True).start()
        return audio_q, transcript_q


    # ── launch the first recogniser as soon as the socket opens --------------
    audio_q, transcript_q = launch_recogniser()

    # ── MAIN LOOP ------------------------------------------------------------
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            # push EVERY inbound PCM frame into *current* queue
            if msg.get("text"):
                payload = json.loads(msg["text"])
                if payload.get("kind") == "AudioData":
                    pcm = base64.b64decode(payload["audioData"]["data"])
                    audio_q.put(pcm)

                    if participant_raw_id is None:
                        participant_raw_id = payload["audioData"]["participantRawID"]

                        # ── PLAY WELCOME GREETING only once ────────────────────────────────
                        if not call_state[call_id]["greeted"]:
                            greeting_pcm  = synthesize(welcome_text)
                            cancel_event  = asyncio.Event()            # we’ll cancel if caller barges in
                            play_task = asyncio.create_task(
                                send_pcm_over_ws(
                                    ws, greeting_pcm, audio_q, participant_raw_id, cancel_event
                                )
                            )
                            call_state[call_id].update({
                                "greeted":     True,
                                "is_playing":  True,
                                "play_task":   play_task,
                                "cancel_evt":  cancel_event,
                            })
                            call_state[call_id]["history"].append(
                                {"role": "assistant", "content": welcome_text}
                            )


            # handle transcripts without blocking
            while not transcript_q.empty():
                is_final, text = transcript_q.get()
                print(text)
                if not text:
                    continue

                if not is_final and call_state[call_id]["is_playing"]:
                    # 1) raise the flag so `send_pcm_over_ws` stops sending
                    call_state[call_id]["cancel_evt"].set()

                    # 2) wait for the task to finish flushing / StopAudio
                    await call_state[call_id]["play_task"]

                    # 3) update state
                    call_state[call_id]["is_playing"] = False
                    continue


                if is_final and text:
                    # 1) append caller utterance to history -------------------------------
                    call_state[call_id]["history"].append(
                        {"role": "user", "content": text}
                    )

                    # 2) get assistant reply ---------------------------------------------
                    # reply_text = gpt_answer(call_state[call_id]["history"])
                    reply_text = gemini_answer(text)
                    reply_pcm  = synthesize(reply_text)

                    # 3) append assistant reply to history --------------------------------
                    call_state[call_id]["history"].append(
                        {"role": "assistant", "content": reply_text}
                    )

                    # 4) create cancel flag & stream the reply (unchanged) ----------------
                    cancel_event = asyncio.Event()
                    play_task = asyncio.create_task(
                        send_pcm_over_ws(
                            ws, reply_pcm, audio_q, participant_raw_id, cancel_event
                        )
                    )

                    call_state[call_id].update(
                        {"is_playing": True,
                        "play_task":  play_task,
                        "cancel_evt": cancel_event}
                    )



    except Exception as e:
        print("WebSocket closed:", e)
    finally:
        audio_q.put(None)
        await ws.close()
# ───────────────────────────── MP3 serving ─────────────────────────────────────
@app.get("/audio/{clip_id}")
def audio(clip_id: str):
    if clip_id in audio_cache:
        return Response(content=audio_cache[clip_id][0], media_type="audio/mpeg")
    return Response(status_code=404)

# ───────────────────────────── dev helper ──────────────────────────────────────
@app.post("/test/")
def test(text: str = Form(...)):
    ans   = gpt_answer(text)
    url   = synthesize(ans)
    return {"reply": ans, "audio_url": url}

# ───────────────────────────── run ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("inbound_main:app", host="0.0.0.0", port=int(os.getenv("PORT", 5503)))
