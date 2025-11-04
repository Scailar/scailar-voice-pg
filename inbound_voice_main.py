# main.py  (complete MVP)
from urllib.parse import urlencode
import os, uuid, time, threading, requests, uvicorn
from fastapi import FastAPI, Request, Response, Form
from google.cloud import texttospeech
from azure.communication.callautomation import (
    CallAutomationClient,
    RecognizeInputType,
    PhoneNumberIdentifier,
    FileSource            # we'll point it at the Google-TTS MP3
)
from pydantic import BaseModel

# ─── environment ────────────────────────────────────────────────────────────────
ACS_CONN_STR           = os.getenv("ACS_CONNECTION_STRING")
ACS_PUBLIC_BASE        = os.getenv("ACS_PUBLIC_BASE")          # e.g. https://mybot.azurewebsites.net
AZ_OPENAI_KEY          = os.getenv("AZURE_OPENAI_SERVICE_KEY")
AZ_OPENAI_ENDPOINT     = os.getenv("AZURE_OPENAI_SERVICE_ENDPOINT")
AZ_OPENAI_DEPLOYMENT   = os.getenv("AZURE_OPENAI_DEPLOYMENT_MODEL_NAME")
AZ_OPENAI_DEPLOYMENT_VERSION   = os.getenv("AZURE_OPENAI_DEPLOYMENT_VERSION")

call_client = CallAutomationClient.from_connection_string(ACS_CONN_STR)
gcp_tts     = texttospeech.TextToSpeechClient()

app = FastAPI()
audio_cache = {}  # {id: (bytes, timestamp)}

# ─── tiny in-memory cache with expiry ───────────────────────────────────────────
def _clean_cache():
    while True:
        now = time.time()
        for k, (_, ts) in list(audio_cache.items()):
            if now - ts > 120:
                audio_cache.pop(k, None)
        time.sleep(30)
threading.Thread(target=_clean_cache, daemon=True).start()

def synthesize(text:str) -> str:
    """Call Google TTS and store result in RAM; return a public URL."""
    audio_cfg = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    voice     = texttospeech.VoiceSelectionParams(language_code="de-DE", name="de-DE-Chirp3-HD-Fenrir")
    resp      = gcp_tts.synthesize_speech(input=texttospeech.SynthesisInput(text=text), voice=voice, audio_config=audio_cfg)
    clip_id   = str(uuid.uuid4())
    audio_cache[clip_id] = (resp.audio_content, time.time())
    return f"{ACS_PUBLIC_BASE}/audio/{clip_id}"

def chat_with_gpt(user_text:str)->str:
    payload = {
        "messages":[
            {"role":"system","content":"You are a helpful voice assistant speaking German."},
            {"role":"user","content":user_text}
        ],
        "max_tokens":200,
        "temperature":0.7,
        "top_p":0.95
    }
    url = f"{AZ_OPENAI_ENDPOINT}/openai/deployments/{AZ_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZ_OPENAI_DEPLOYMENT_VERSION}"
    r   = requests.post(url, json=payload, headers={"api-key":AZ_OPENAI_KEY})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ─── event models (partial) ─────────────────────────────────────────────────────
class IncomingCall(BaseModel):
    incomingCallContext:str
    callerId:str
    from_:str|None = None
    to:str|None     = None
    # ... other ACS fields omitted

@app.post("/incoming-call/")
async def incoming_call(request: Request):
    """
    Handles BOTH the initial Event Grid handshake **and**
    real IncomingCall events from Azure Communication Services.
    """
    body = await request.json()
    if body[0]["data"]["to"]["phoneNumber"]["value"] != "+4961171186831":
        return

    # 1️⃣ — Event Grid validation ping
    #    Azure wraps the event(s) in *a list*, even for a single item.
    if isinstance(body, list) and body and \
       body[0].get("eventType") == "Microsoft.EventGrid.SubscriptionValidationEvent":
        validation_code = body[0]["data"]["validationCode"]
        # Azure expects status 200 and the code in 'validationResponse'
        return {"validationResponse": validation_code}

    # 2️⃣ — Normal IncomingCall event (original logic)
    evt = body[0]["data"]          # pydantic model from the full MVP
    query_parameters = urlencode(
                            {"callerId": body[0]["data"]["from"]["phoneNumber"]["value"]}
                        )
    cognitive_service_endpoint: str = os.getenv("COGNITIVE_SERVICE_ENDPOINT")
    call = call_client.answer_call(
        incoming_call_context=evt["incomingCallContext"],
        cognitive_services_endpoint=cognitive_service_endpoint,
        callback_url=f"{ACS_PUBLIC_BASE}/acs-events/?{query_parameters}"
    )
    # call_client.get_call_connection(call.call_connection_id)\
    #            .start_continuous_recognition()

    return {"status": "answered"}


@app.post("/acs-events/")
async def acs_events(callerId, req:Request):
    # caller_id = req.args.get("callerId")
    body = await req.json()
    for ev in body:
        etype = ev.get("type")
        data = ev["data"]
        conn  = data.get("callConnectionId")
        if etype == "recognitionStarted":
            pass  # could update in-memory FSM
        elif etype == "Microsoft.Communication.RecognizeCompleted":
            text = data.get("speechResult",{}).get("speech")
            print(text)
            reply = chat_with_gpt(text)
            audio_url = synthesize(reply)
            prompt_src = FileSource(url=audio_url)
            call_conn = call_client.get_call_connection(ev["data"]["callConnectionId"])
            call_conn.start_recognizing_media(
                input_type=RecognizeInputType.SPEECH,  # Expect speech input
                target_participant=PhoneNumberIdentifier(callerId),
                end_silence_timeout=1,  # End recognition after 2 seconds of silence
                play_prompt=prompt_src,  # Play prompt to the caller
                # operation_context=context,
                speech_language="de-DE",
                interrupt_call_media_operation=True,
            )
            # if data.get("speechResult",{}).get("speech"):
                # user spoke while we were playing → stop playback = barge-in
                # call_client.get_call_connection(conn).cancel_all_media_operations()

        elif etype == "Microsoft.Communication.CallConnected":           # <-- stays the same
            # text = ev["recognitionResult"]["text"].strip()
            # if not text:
            #     continue

            # reply = chat_with_gpt(text)
            audio_url = synthesize("Hallo, wie kann ich Ihnen helfen?")        # Google TTS → returns https://.../audio/{id}

            # 👉 instead of call_connection.play_audio_from_url(...)
            #    build a "prompt + recognize" request
            prompt_src = FileSource(url=audio_url)

            call_conn = call_client.get_call_connection(ev["data"]["callConnectionId"])

            call_conn.start_recognizing_media(
                input_type=RecognizeInputType.SPEECH,  # Expect speech input
                target_participant=PhoneNumberIdentifier(callerId),
                end_silence_timeout=1,  # End recognition after 2 seconds of silence
                play_prompt=prompt_src,  # Play prompt to the caller
                # operation_context=context,
                speech_language="de-DE",
                interrupt_call_media_operation=True,
            )

            # call_conn.start_recognizing_media(
            #     input_type = RecognizeInputType.SPEECH,
            #     recognize_options = opts
            # )
    return {"ok":True}

@app.get("/audio/{clip_id}")
def audio(clip_id:str):
    if clip_id in audio_cache:
        return Response(content=audio_cache[clip_id][0], media_type="audio/mpeg")
    return Response(status_code=404)

@app.post("/test/")
def test(text:str=Form(...)):
    reply = chat_with_gpt(text)
    url   = synthesize(reply)
    return {"reply":reply, "audio_url":url}

if __name__ == "__main__":
    uvicorn.run("inbound_voice_main:app", host="0.0.0.0", port=5503)
