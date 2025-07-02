# main.py  ── FastAPI backend for Calendar‑Booking bot
import os, json, logging, importlib.util
from datetime import datetime, timedelta
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_groq.chat_models import ChatGroq

from google.oauth2 import service_account
from googleapiclient.discovery import build

from dotenv import load_dotenv
load_dotenv()  # → loads .env when you run locally

# ────────────────────────────── Logging ──────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────── Helpers for credentials ─────────────────────
def _get_streamlit_secrets():
    """
    Import streamlit *dynamically* so local uvicorn runs don’t
    require Streamlit in the venv. Returns st.secrets or {}.
    """
    if importlib.util.find_spec("streamlit"):
        import streamlit as st
        return getattr(st, "secrets", {})
    return {}

def _load_google_credentials() -> service_account.Credentials | None:
    """Return google service‑account creds or None."""
    scopes = ["https://www.googleapis.com/auth/calendar"]
    st_secrets = _get_streamlit_secrets()

    # 1️⃣ Streamlit Cloud / secrets.toml
    try:
        creds_toml = st_secrets.get("google", {}).get("credentials")
        if creds_toml:
            logger.info("Loaded Google credentials from st.secrets")
            return service_account.Credentials.from_service_account_info(
                json.loads(creds_toml), scopes=scopes
            )
    except Exception as e:
        logger.warning(f"Could not read creds from st.secrets: {e}")

    # 2️⃣ Environment variable with raw JSON
    if "GOOGLE_SERVICE_ACCOUNT_JSON" in os.environ:
        try:
            logger.info("Loaded Google credentials from env var GOOGLE_SERVICE_ACCOUNT_JSON")
            return service_account.Credentials.from_service_account_info(
                json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]), scopes=scopes
            )
        except Exception as e:
            logger.warning(f"GOOGLE_SERVICE_ACCOUNT_JSON invalid: {e}")

    # 3️⃣ Credentials file on disk
    default_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
    if os.path.exists(default_path):
        try:
            logger.info(f"Loaded Google credentials from file {default_path}")
            return service_account.Credentials.from_service_account_file(
                default_path, scopes=scopes
            )
        except Exception as e:
            logger.warning(f"credentials.json invalid: {e}")

    # Fallback → no creds
    return None


# ─────────────────────── Google Calendar client ──────────────────────
CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")
credentials = _load_google_credentials()

if credentials:
    calendar_service = build("calendar", "v3", credentials=credentials)
else:
    calendar_service = None
    logger.error(
        "Google Calendar credentials not found. "
        "Availability & booking tools will reply with an error message."
    )

# ────────────────── Calendar helper functions / tools ─────────────────
def _no_service_msg() -> str:
    return (
        "Google Calendar is not configured (missing credentials). "
        "Please contact the administrator."
    )

def check_availability(date: str) -> str:
    """Tool: check all events on a given date (YYYY‑MM‑DD, UTC)."""
    if not calendar_service:
        return _no_service_msg()
    try:
        events = (
            calendar_service.events()
            .list(
                calendarId=CALENDAR_ID,
                timeMin=f"{date}T00:00:00Z",
                timeMax=f"{date}T23:59:59Z",
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
            .get("items", [])
        )
        if not events:
            return f"All slots are free on {date}."
        booked = [
            f"{e['start'].get('dateTime', e['start'].get('date'))} — {e.get('summary', 'No title')}"
            for e in events
        ]
        return "Booked slots on {0}: {1}".format(date, "; ".join(booked))
    except Exception as e:
        logger.error(f"check_availability: {e}")
        return f"Failed to check availability: {e}"

def suggest_slots(date_range: str) -> str:
    """
    Tool: check free/busy between two dates. `date_range` format
    → 'YYYY‑MM‑DD to YYYY‑MM‑DD'
    """
    if not calendar_service:
        return _no_service_msg()
    try:
        start, end = [d.strip() for d in date_range.split("to")]
        fb = (
            calendar_service.freebusy()
            .query(
                body={
                    "timeMin": f"{start}T00:00:00Z",
                    "timeMax": f"{end}T23:59:59Z",
                    "items": [{"id": CALENDAR_ID}],
                }
            )
            .execute()
        )
        busy = fb["calendars"][CALENDAR_ID]["busy"]
        if not busy:
            return f"All slots are free from {start} to {end}."
        busy_str = "; ".join(f"{b['start']} → {b['end']}" for b in busy)
        return f"Busy slots: {busy_str}"
    except Exception as e:
        logger.error(f"suggest_slots: {e}")
        return f"Failed to suggest slots: {e}"

def book_appointment(input_str: str) -> str:
    """
    Tool: create a 1‑hour event. Accepts either
    '2025‑07‑06T14:00:00 John'  or  '2025‑07‑06T14:00:00,John'
    (UTC time).
    """
    if not calendar_service:
        return _no_service_msg()
    try:
        date_time, user_name = (
            [x.strip() for x in input_str.split(",", 1)]
            if "," in input_str
            else input_str.strip().split(" ", 1)
        )
        start_dt = datetime.fromisoformat(date_time.replace("Z", "+00:00"))
        end_dt = start_dt + timedelta(hours=1)

        event = {
            "summary": f"Appointment with {user_name}",
            "start": {"dateTime": start_dt.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": "UTC"},
        }
        created = (
            calendar_service.events()
            .insert(calendarId=CALENDAR_ID, body=event)
            .execute()
        )
        return f"Booked {user_name} at {date_time}. Link: {created.get('htmlLink')}"
    except Exception as e:
        logger.error(f"book_appointment: {e}", exc_info=True)
        return f"Failed to book appointment: {e}"

# ─────────────────── LangChain tools & conversational agent ───────────
tools = [
    Tool(
        name="check_availability",
        func=check_availability,
        description="Check calendar availability for a single date (YYYY‑MM‑DD, UTC).",
    ),
    Tool(
        name="suggest_slots",
        func=suggest_slots,
        description="Suggest free/busy slots between two dates: 'YYYY‑MM‑DD to YYYY‑MM‑DD'.",
    ),
    Tool(
        name="book_appointment",
        func=book_appointment,
        description="Book a 1‑hour appointment: 'YYYY‑MM‑DDTHH:MM:SS Name' or with a comma.",
    ),
]

# ─────────────────────────── Groq LLM setup ───────────────────────────
st_secrets = _get_streamlit_secrets()
GROQ_API_KEY = (
    st_secrets.get("GROQ_API_KEY")
    or os.getenv("GROQ_API_KEY")
)

if not GROQ_API_KEY:
    # Warn but don’t crash the container – FastAPI will return 500 if the user hits /chat
    logger.error("GROQ_API_KEY not found in secrets or env vars.")

llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY,
    temperature=0.2,
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(
    tools, llm, agent="chat-conversational-react-description",
    memory=memory, verbose=True
)

# ────────────────────────── FastAPI endpoints ─────────────────────────
app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")
    try:
        return ChatResponse(response=agent.run(req.message))
    except Exception as e:
        logger.error("/chat error", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

# ─────────────── optional basic routes & in‑memory store ──────────────
class MsgPayload(BaseModel):
    msg_id: int
    msg_name: str

messages_list: Dict[int, MsgPayload] = {}

@app.get("/")
def root():
    return {"message": "Hello from the Calendar‑Booking bot 👋"}

@app.get("/about")
def about():
    return {"message": "Conversational AI Appointment Booking Assistant"}

@app.post("/messages/")
def add_msg(payload: MsgPayload):
    messages_list[payload.msg_id] = payload
    return {"message": payload}

@app.get("/messages")
def message_items():
    return {"messages": messages_list}
