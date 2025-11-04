import sys
import os
import json
import base64
import asyncio
import datetime
import uuid
import audioop
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Any, List

import uvicorn
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

# Azure Communication Services -------------------------------------------------
from azure.communication.callautomation import (
    CallAutomationClient,
    PhoneNumberIdentifier,
    MediaStreamingOptions,
    MediaStreamingTransportType,
    MediaStreamingContentType,
    MediaStreamingAudioChannelType,
    AudioFormat,
)

# Google Gemini ----------------------------------------------------------------
from google import genai
from google.genai import types
import datetime

# MS Graph (Outlook) -----------------------------------------------------------
try:
    import msal
    import requests
except ImportError:  # pragma: no cover – optional dependency
    msal = None  # type: ignore
    requests = None  # type: ignore

# ─── Logging setup ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "server.log")),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("scailar")
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").disabled = True
logging.getLogger("urllib3.connectionpool").disabled = True
logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("websockets.client").disabled = True
logging.getLogger("google_genai.live").disabled = True
logging.getLogger("google_genai.types").disabled = True

# ─── ENV ----------------------------------------------------------------------
ACS_CONN = os.getenv("ACS_CONNECTION_STRING")
ORIG_PHONE = "+replace"
BASE_HTTP = os.getenv("ACS_PUBLIC_BASE")

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# MS GRAPH CREDS ---------------------------------------------------------------
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
OUTLOOK_CALENDAR_ID = os.getenv("OUTLOOK_CALENDAR_ID")  # default calendar if absent

# ─── Gemini config ────────────────────────────────────────────────────────────
MODEL = "models/gemini-2.0-flash-live-001"

end_call_tool = {
    "name": "end_call",
    "description": "Terminates the current call immediately.",
    "parameters": {"type": "object"},
}

make_appt_tool = {
    "name": "make_appointment",
    "description": (
        "Creates an appointment in Outlook for the caller. "
        "All times are assumed to be in Europe/Berlin unless explicitly specified."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "Appointment subject/title."},
            "start_time_iso": {
                "type": "string",
                "description": "Start time in ISO‑8601 format, e.g. 2025-05-12T14:00:00+02:00",
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Duration of the meeting in minutes (default 30).",
                "minimum": 5,
                "maximum": 480,
            },
            "attendee_email": {
                "type": "string",
                "description": "E‑mail of the primary attendee (caller).",
            },
        },
        "required": ["subject", "start_time_iso", "attendee_email"],
    },
}

TOOLS = [{"function_declarations": [end_call_tool, make_appt_tool]}]

LIVE_CFG = types.LiveConnectConfig(
    response_modalities=["audio"],
    output_audio_transcription={},
    input_audio_transcription={},
    temperature=0.1,
    tools=TOOLS,
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        ),
        language_code="de-DE",
    ),
    system_instruction=types.Content(
        parts=[
            types.Part.from_text(
                text="""
Handle den folgenden Anruf ausschließlich auf Deutsch.

Rolle und Kontext:
Du bist der freundliche und professionelle KI-Agent von Scailar, einem Anbieter für KI-basierte Lösungen speziell für Arztpraxen. Dieser Anruf dient als Demo-Test, bei dem der Anrufer die Fähigkeiten von Scailar testen möchte.

Gesprächsablauf:

Der Anruf beginnt mit einer Systemnachricht, die Metadaten des Anrufers enthält.
Deine erste Aufgabe ist es, den Anrufer freundlich zu begrüßen und ihn zu fragen, mit wem du sprichst. 
Nachdem der Anrufer seinen Namen genannt hat, gib eine kurze, natürliche und professionelle Einleitung. Erkläre, dass du der KI-Agent von Scailar bist und hier bist, um ihm unsere Lösungen für Arztpraxen in dieser Demo zu zeigen.
Fahre direkt im Anschluss mit einer konkreten Frage fort, um herauszufinden, welchen Bereich der Anrufer in dieser Demo näher kennenlernen möchte oder welche Herausforderung in seiner Praxis ihn am meisten interessiert. Verwende dabei nicht explizit das Wort "Szenario".
Reaktion auf Anruferinteresse:

Deine Hauptaufgabe ist es, den Anrufer durch eines der folgenden Themen zu führen, basierend auf seinem Interesse:

Fokus auf Anwendung/Lösung: Wenn der Anrufer Interesse an konkreten Vorteilen oder Anwendungsfällen zeigt (z.B. Reduzierung von Ausfallrate bei Terminen, Terminautomatisierung, Integration), erkläre kurz, dass Scailar's telefonischer KI-Agent genau dabei helfen kann, die Ausfallrate deutlich zu reduzieren, die Terminvereinbarung zu automatisieren und sich nahtlos in das Praxisverwaltungssystem (PVS) integrieren lässt. 
Fokus auf Hintergrund/Team: Wenn der Anrufer ein persönliches Gespräch mit dem Entwicklungsteam oder generell "mehr Informationen" wünscht, frage höflich, ob wir ihn unter der aktuellen Telefonnummer für ein solches Gespräch kontaktieren können.
Wichtige Anweisungen:

Sprich durchgehend klar, höflich, professionell und engagiert.
Gib niemals medizinische Beratung oder rechtlich bindende Zusagen.
Leite das Gespräch nicht weiter, sondern sammele nur die Informationen über den Anrufer.
Stelle pro Runde immer nur eine Frage an den Anrufer.
"""
            )
        ],
        role="user",
    ),
)
#  oder ob der Anrufer direkt einen Termin vereinbaren möchte. Für eine direkte Vereinbarung, nutze die Funktion make_appointment und stelle sicher, dass die Email korrekt ist.

GREETING = "Hallo"

# ─── Audio constants ───────────────────────────────────────────────────────────
IN_RATE = 16_000
OUT_RATE = 24_000
FRAME_MS = 30
FRAME_LEN_IN = IN_RATE * 2 * FRAME_MS // 1000
FRAME_LEN_OUT = OUT_RATE * 2 * FRAME_MS // 1000

# ─── Helper functions ─────────────────────────────────────────────────────────


def down24k_to_16k(buf: bytes) -> bytes:
    return audioop.ratecv(buf, 2, 1, OUT_RATE, IN_RATE, None)[0]


def ts() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


# ─── Call tracking state ──────────────────────────────────────────────────────


@dataclass
class CallState:
    caller_id: str
    call_id: str
    started: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    state: str = "dialing"
    conversation_log: List[Dict[str, Any]] = field(default_factory=list)
    input_text: str = ""
    output_text: str = ""
    completed: bool = False

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data["started"] = self.started.isoformat()
        return data


class CallRegistry:
    """Thread‑safe registry for active + finished calls."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._calls: Dict[str, CallState] = {}

    async def add(self, call: CallState) -> None:
        async with self._lock:
            self._calls[call.call_id] = call
            logger.info("[CALL %s] added – caller=%s", call.call_id, call.caller_id)

    async def update(self, call_id: str, **kwargs) -> None:
        async with self._lock:
            call = self._calls.get(call_id)
            if not call:
                return
            for k, v in kwargs.items():
                setattr(call, k, v)

    async def get(self, call_id: str) -> Optional[CallState]:
        async with self._lock:
            return self._calls.get(call_id)

    async def list(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return [c.to_json() for c in self._calls.values()]


registry = CallRegistry()

# ─── Outlook integration ──────────────────────────────────────────────────────


def _get_graph_client() -> Optional[Any]:  # type: ignore
    if not all([AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID]):
        logger.warning("Graph credentials missing; appointment creation disabled.")
        return None
    if msal is None:
        logger.error("msal library not installed; install azure‑identity & msal.")
        return None

    app = msal.ConfidentialClientApplication(
        AZURE_CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{AZURE_TENANT_ID}",
        client_credential=AZURE_CLIENT_SECRET,
    )
    token = app.acquire_token_for_client(["https://graph.microsoft.com/.default"])
    if "access_token" not in token:
        logger.error(
            "Unable to acquire Graph token: %s", token.get("error_description")
        )
        return None
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token['access_token']}",
            "Content-Type": "application/json",
        }
    )
    return session


def schedule_outlook_appointment(
    subject: str,
    start_time_iso: str,
    duration_minutes: int,
    attendee_email: str,
) -> str:
    """Create an Outlook event and return its iCal UID or link."""
    logger.info(
        f"Creating appointment with subject {subject}, start time {start_time_iso}, attendee email {attendee_email}"
    )
    session = _get_graph_client()
    if session is None:
        raise RuntimeError("Outlook scheduling not configured.")

    start_dt = datetime.datetime.fromisoformat(start_time_iso)
    end_dt = start_dt + datetime.timedelta(minutes=duration_minutes or 30)

    event = {
        "subject": subject,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": "Europe/Berlin"},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": "Europe/Berlin"},
        "location": {"displayName": "Telefontermin"},
        "attendees": [
            {
                "emailAddress": {"address": attendee_email, "name": attendee_email},
                "type": "required",
            }
        ],
        "allowNewTimeProposals": True,
        "isOnlineMeeting": True,
        "onlineMeetingProvider": "teamsForBusiness",
    }

    cal_id = OUTLOOK_CALENDAR_ID or "primary"
    resp = session.post(
        f"https://graph.microsoft.com/v1.0/users/{cal_id}/calendar/events",
        data=json.dumps(event),
    )
    if resp.status_code >= 300:
        logger.error("Graph create event failed: %s – %s", resp.status_code, resp.text)
        raise RuntimeError("Outlook appointment could not be created.")
    data = resp.json()
    logger.info(
        "Outlook appointment created – subject='%s' id=%s", subject, data.get("id")
    )
    return data.get("id", "")


# ─── FastAPI application ─────────────────────────────────────────────────────-
app = FastAPI()
call_client = CallAutomationClient.from_connection_string(ACS_CONN)


# ─── Utility ──────────────────────────────────────────────────────────────────
async def end_call(call_id: str):
    call_state = await registry.get(call_id)
    if not call_state:
        return
    logger.info("[CALL %s] Ending call", call_id)
    conn = call_client.get_call_connection(call_id)
    await conn.hang_up(is_for_everyone=True)
    await registry.update(call_id, completed=True, state="ended")


# ─── REST endpoints ─────────────────────────────────────────────────────────--


@app.post("/dial")
async def dial(req: Request):
    body = await req.json()
    to_number: str = body.get("to")
    if not to_number:
        raise HTTPException(400, detail="Missing 'to' number")

    logger.info("Dialing %s", to_number)

    ws_url = f"wss://{BASE_HTTP.replace('https://', '')}/media"

    opts = MediaStreamingOptions(
        transport_url=ws_url,
        transport_type=MediaStreamingTransportType.WEBSOCKET,
        content_type=MediaStreamingContentType.AUDIO,
        audio_channel_type=MediaStreamingAudioChannelType.UNMIXED,
        audio_format=AudioFormat.PCM16_K_MONO,
        start_media_streaming=True,
        enable_bidirectional=True,
    )

    call = call_client.create_call(
        target_participant=[PhoneNumberIdentifier(to_number)],
        source_caller_id_number=PhoneNumberIdentifier(ORIG_PHONE),
        callback_url=f"{BASE_HTTP}/events",
        media_streaming=opts,
    )

    state = CallState(caller_id=to_number, call_id=call.call_connection_id)
    await registry.add(state)

    return {"call_id": call.call_connection_id}


@app.post("/events")
async def events(req: Request):
    evs = await req.json()
    if evs[0].get("eventType") == "Microsoft.EventGrid.SubscriptionValidationEvent":
        return {"validationResponse": evs[0]["data"]["validationCode"]}

    for ev in evs:
        cid = ev["data"].get("callConnectionId")
        if not cid:
            continue
        if ev["type"] == "Microsoft.Communication.CallConnected":
            await registry.update(cid, state="connected")
        elif ev["type"] == "Microsoft.Communication.CallDisconnected":
            await registry.update(cid, state="disconnected", completed=True)
    return {"ok": True}


@app.get("/calls")
async def list_calls():
    """Return lightweight view of current + past calls."""
    return JSONResponse(content=await registry.list())


# ─── Bidirectional media websocket ─────────────────────────────────────────---
@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    cid = ws.headers.get("x-ms-call-connection-id")
    if not cid:
        await ws.close(code=4001)
        return

    call_state = await registry.get(cid)
    if not call_state:
        await ws.close(code=4002)
        return

    part_id: Optional[str] = None
    in_q: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue(maxsize=100)
    out_q: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue(maxsize=100)

    async def gemini_loop():
        client = genai.Client(
            api_key=GEMINI_KEY, http_options={"api_version": "v1beta"}
        )
        async with client.aio.live.connect(model=MODEL, config=LIVE_CFG) as session:
            await session.send(
                input=f"Telefonnummer: {call_state.caller_id}; aktuelles Datum: {str(datetime.date.today())}",
                end_of_turn=True,
            )

            async def feed():
                while True:
                    chunk = await in_q.get()
                    if chunk is None:
                        break
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                    )

            async def drain():
                while True:
                    turn = session.receive()
                    async for part in turn:
                        if part.data:
                            await out_q.put(part.data)
                        sc = getattr(part, "server_content", None)
                        if sc and sc.output_transcription:
                            call_state.output_text += sc.output_transcription.text
                        if sc and sc.input_transcription:
                            call_state.input_text += sc.input_transcription.text
                        if part.tool_call:
                            responses = []
                            for fc in part.tool_call.function_calls:
                                if fc.name == "end_call":
                                    await end_call(cid)
                                    responses.append(
                                        types.FunctionResponse(
                                            id=fc.id,
                                            name=fc.name,
                                            response={"result": "call ending"},
                                        )
                                    )
                                elif fc.name == "make_appointment":
                                    try:
                                        appt_id = schedule_outlook_appointment(
                                            subject=fc.args["subject"],
                                            start_time_iso=fc.args["start_time_iso"],
                                            duration_minutes=fc.args.get(
                                                "duration_minutes", 30
                                            ),
                                            attendee_email=fc.args["attendee_email"],
                                        )
                                        responses.append(
                                            types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response={"appointment_id": appt_id},
                                            )
                                        )
                                    except Exception as ex:
                                        logger.exception("Failed to create appointment")
                                        responses.append(
                                            types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response={"error": str(ex)},
                                            )
                                        )
                            await session.send_tool_response(
                                function_responses=responses
                            )

            async with asyncio.TaskGroup() as tg:
                tg.create_task(feed())
                tg.create_task(drain())

    async def play_to_acs():
        buffer: List[bytes] = []
        buffer_size = 5
        while True:
            data24 = await out_q.get()
            if data24 is None:
                break
            pcm16 = down24k_to_16k(data24)
            for i in range(0, len(pcm16), FRAME_LEN_IN):
                frame = pcm16[i : i + FRAME_LEN_IN]
                buffer.append(frame)
                if len(buffer) >= buffer_size:
                    for bf in buffer:
                        payload = {
                            "kind": "AudioData",
                            "audioData": {
                                "timestamp": ts(),
                                "participantRawID": part_id or "",
                                "data": base64.b64encode(bf).decode(),
                                "silent": False,
                            },
                        }
                        await ws.send_text(json.dumps(payload))
                        await asyncio.sleep(FRAME_MS / 1000)
                    buffer.clear()
        for bf in buffer:
            payload = {
                "kind": "AudioData",
                "audioData": {
                    "timestamp": ts(),
                    "participantRawID": part_id or "",
                    "data": base64.b64encode(bf).decode(),
                    "silent": False,
                },
            }
            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(FRAME_MS / 1000)

    async def pull_from_acs():
        nonlocal part_id
        while True:
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            if msg["type"] == "websocket.disconnect":
                break
            if text := msg.get("text"):
                packet = json.loads(text)
                if packet.get("kind") != "AudioData":
                    continue
                meta = packet["audioData"]
                if part_id is None:
                    part_id = meta["participantRawID"]
                pcm = base64.b64decode(meta["data"])
                await in_q.put(pcm)

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(gemini_loop())
            tg.create_task(play_to_acs())
            tg.create_task(pull_from_acs())
    finally:
        await in_q.put(None)
        await out_q.put(None)
        await registry.update(cid, completed=True)
        logger.info(
            "[CALL %s] Completed – in='%s' out='%s'",
            cid,
            call_state.input_text,
            call_state.output_text,
        )
        try:
            await ws.close()
        except Exception:
            pass


# ─── Runner ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("outbound_live_main:app", host="0.0.0.0", port=5503, reload=False)
