# outbound_main.py  ── all-Google speech, ACS telephony (outbound version)
import base64, datetime, json, os, queue, threading, time, uuid, asyncio
import urllib.parse as up
from fastapi import FastAPI, Request, WebSocket, Response
from google.cloud import texttospeech, speech_v1p1beta1 as speech
from google import genai
from google.genai import types
from azure.communication.callautomation import (
    CallAutomationClient, PhoneNumberIdentifier,
    FileSource, MediaStreamingOptions, MediaStreamingTransportType,
    MediaStreamingContentType, MediaStreamingAudioChannelType, AudioFormat
)
import uvicorn

# ─────────── ENV ──────────────────────────────────────────────────────────────
ACS_CONN_STR  = os.getenv("ACS_CONNECTION_STRING")
ORIG_PHONE    = "+replace"      # ACS number to show
PUBLIC_BASE   = os.getenv("ACS_PUBLIC_BASE")               # https://<ngrok>
GEMINI_API_KEY= os.getenv("GEMINI_API_KEY")

# ─────────── CLIENTS ──────────────────────────────────────────────────────────
call_client  = CallAutomationClient.from_connection_string(ACS_CONN_STR)
tts_client   = texttospeech.TextToSpeechClient()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chat = client.chats.create(model="gemini-2.0-flash", config=types.GenerateContentConfig(
        system_instruction="Du bist eine hilfreiche Call Center Agentin im St. Joseph Hospital in Wiesbaden, die Anrufe tätigt, um den Gesundheitsstatus der Patienten abzufragen. Du fängst mit einer Willkommensnachricht an, in der du am Ende die Frage stellst, wie dem Patienten heute geht. Die erste Nachricht des Nutzers ist eine Antwort darauf."),)

# ─────────── FASTAPI APP ──────────────────────────────────────────────────────
app = FastAPI()
FRAME_MS  = 100
FRAME_LEN = 16000 * 2 * FRAME_MS // 1000
SILENCE   = b"\x00" * FRAME_LEN
calls     = {}      # call_id ➜ dict(state)

# ─────────── HELPERS ──────────────────────────────────────────────────────────
def synthesize(text: str) -> bytes:
    voice = texttospeech.VoiceSelectionParams(
        language_code="de-DE", name="de-DE-Chirp3-HD-Achernar"
    )
    audio_cfg = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
    )
    resp = tts_client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=voice, audio_config=audio_cfg
    )
    return resp.audio_content

def gemini_answer(prompt: str) -> str:
    response = chat.send_message(prompt)
    return response.text.strip()

async def send_pcm(ws: WebSocket, pcm: bytes, audio_q, part_id: str,
                   cancel_evt: asyncio.Event):
    ts0 = datetime.datetime.utcnow()
    for i in range(0, len(pcm), FRAME_LEN):
        if cancel_evt.is_set():
            break
        frame = pcm[i:i+FRAME_LEN]
        stamp = (ts0+datetime.timedelta(milliseconds=i//32)).isoformat(timespec="milliseconds")+'Z'
        await ws.send_text(json.dumps({
            "Kind":"AudioData",
            "AudioData":{
                "Timestamp":stamp,
                "ParticipantRawID":part_id,
                "Data":base64.b64encode(frame).decode(),
                "Silent":False
            },
            "StopAudio":None
        }))
        audio_q.put(SILENCE)
        await asyncio.sleep(FRAME_MS/1000)

async def stop_audio(ws:WebSocket):
    await ws.send_text(json.dumps({"Kind":"StopAudio","AudioData":None,"StopAudio":{}}))

# ─────────── REST ENDPOINT TO PLACE CALL ──────────────────────────────────────
@app.post("/dial")
async def dial(req: Request):
    body = await req.json()
    to_number = body["to"]
    ws_url=f"ws://a521-37-201-116-121.ngrok-free.app/media"
    opts=MediaStreamingOptions(
                transport_url=ws_url,
                transport_type=MediaStreamingTransportType.WEBSOCKET,
                content_type=MediaStreamingContentType.AUDIO,
                audio_channel_type=MediaStreamingAudioChannelType.UNMIXED,
                audio_format=AudioFormat.PCM16_K_MONO,
                start_media_streaming=True,
                enable_bidirectional=True
            )
    call = call_client.create_call(
        target_participant=[PhoneNumberIdentifier(to_number)],
        source_caller_id_number=PhoneNumberIdentifier(ORIG_PHONE),
        callback_url=f"{PUBLIC_BASE}/events",
        media_streaming=opts
    )
    calls[call.call_connection_id] = {"state":"dialing"}
    return {"call_id": call.call_connection_id}

# ─────────── EVENT GRID CALLBACK (connect → start media) ──────────────────────
@app.post("/events")
async def events(req: Request):
    payload = await req.json()
    # Subscription validation
    if isinstance(payload, list) and payload[0].get("eventType")=="Microsoft.EventGrid.SubscriptionValidationEvent":
        return {"validationResponse":payload[0]["data"]["validationCode"]}

    for ev in payload:
        if ev["type"]=="Microsoft.Communication.CallConnected":
            # call_client.get_call_connection(ev["data"]["callConnectionId"]).start_media_streaming()
            pass
            
            
            # call_client.get_call_connection(cid).start_media_streaming(opts)
    return {"ok":True}

# ─────────── MEDIA WEBSOCKET ───────────────────────────────────────────────────
@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    cid = ws.headers["x-ms-call-connection-id"]
    calls[cid].update(
        is_playing=False, participant=None,
        audio_q=queue.Queue(), transcripts=queue.Queue()
    )
    audio_q      = calls[cid]["audio_q"]
    transcript_q = calls[cid]["transcripts"]

    # STT thread -------------------------------------------------------------
    def stt_thread():
        client = speech.SpeechClient()
        cfg    = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000, language_code="de-DE",
            enable_automatic_punctuation=True
        )
        s_cfg  = speech.StreamingRecognitionConfig(
            config=cfg, interim_results=True, single_utterance=False
        )
        def gen():                       # generator of audio
            for chunk in iter(audio_q.get, None):
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
        for resp in client.streaming_recognize(s_cfg, gen()):
            for res in resp.results:
                if res.alternatives:
                    transcript_q.put((res.is_final,res.alternatives[0].transcript.strip()))
    threading.Thread(target=stt_thread, daemon=True).start()

    # play greeting once -----------------------------------------------------
    greeted=False

    try:
        while True:
            msg=await ws.receive()
            if msg["type"]=="websocket.disconnect": break
            if msg.get("text"):
                data=json.loads(msg["text"])
                if data.get("kind")=="AudioData":
                    pcm=base64.b64decode(data["audioData"]["data"])
                    audio_q.put(pcm)
                    if not calls[cid]["participant"]:
                        calls[cid]["participant"]=data["audioData"]["participantRawID"]
                        if not greeted:
                            greet_pcm=synthesize("Hallo, ich bin KI-Telefonassistentin des St. Josefs-Hospitals in Wiesbaden. Ich möchte einige Informationen über Ihre Gesundheit einholen. Zunächst, wie geht es Ihnen?")
                            ce=asyncio.Event(); pt=asyncio.create_task(
                                send_pcm(ws,greet_pcm,audio_q,calls[cid]["participant"],ce))
                            calls[cid].update(is_playing=True,play_task=pt,cancel_evt=ce)
                            greeted=True

            # handle STT results --------------------------------------------
            while not transcript_q.empty():
                is_final,text=transcript_q.get()
                if not text: continue

                # barge-in
                if not is_final and calls[cid]["is_playing"]:
                    calls[cid]["cancel_evt"].set()
                    await calls[cid]["play_task"]
                    calls[cid]["is_playing"]=False
                    continue

                if is_final:
                    print(text)
                    reply=gemini_answer(text)
                    print(reply)
                    reply_pcm=synthesize(reply)
                    ce=asyncio.Event()
                    pt=asyncio.create_task(
                        send_pcm(ws,reply_pcm,audio_q,calls[cid]["participant"],ce))
                    calls[cid].update(is_playing=True,play_task=pt,cancel_evt=ce)
    finally:
        audio_q.put(None)
        await ws.close()

# ─────────── RUN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("outbound_main:app", host="0.0.0.0", port=5503)
