# main.py — FastAPI backend for Calendar-Booking bot

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
load_dotenv()

# ─────────────── Logging ───────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ────── Streamlit secrets helper ──────
def _get_streamlit_secrets():
    if importlib.util.find_spec("streamlit"):
        import streamlit as st
        return getattr(st, "secrets", {})
    return {}

# ────── Load Google credentials ──────
def _load_google_credentials() -> service_account.Credentials | None:
    scopes = ["https://www.googleapis.com/auth/calendar"]
    st_secrets = _get_streamlit_secrets()

    try:
        creds_dict = st_secrets.get("google_credentials")
        if creds_dict:
            logger.info("Loaded Google credentials from st.secrets")
            return service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    except Exception as e:
        logger.warning(f"Could not read creds from st.secrets: {e}")

    if "GOOGLE_SERVICE_ACCOUNT_JSON" in os.environ:
        try:
            logger.info("Loaded Google credentials from env var")
            return service_account.Credentials.from_service_account_info(
                json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]), scopes=scopes
            )
        except Exception as e:
            logger.warning(f"Env var creds invalid: {e}")

    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
    if os.path.exists(path):
        try:
            logger.info(f"Loaded Google credentials from file {path}")
            return service_account.Credentials.from_service_account_file(path, scopes=scopes)
        except Exception as e:
            logger.warning(f"File creds invalid: {e}")

    return None

# ────── Calendar Setup ──────
CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")
credentials = _load_google_credentials()

if credentials:
    calendar_service = build("calendar", "v3", credentials=credentials)
else:
    calendar_service = None
    logger.error("Google Calendar credentials not found. Calendar tools disabled.")

# ────── Google Calendar tools ──────
def _no_service_msg() -> str:
    return "Google Calendar is not configured (missing credentials)."

def check_availability(date: str) -> str:
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
                orderBy="startTime"
            )
            .execute()
            .get("items", [])
        )
        if not events:
            return f"All slots are free on {date}."
        booked = [
            f"{e['start'].get('dateTime', e['start'].get('date'))} — {e.get('summary', 'No title')}"
            for e in events
        ]
        return f"Booked slots on {date}: " + "; ".join(booked)
    except Exception as e:
        logger.error(f"check_availability error: {e}", exc_info=True)
        return f"Error checking availability: {e}"

def suggest_slots(date_range: str) -> str:
    if not calendar_service:
        return _no_service_msg()
    try:
        start, end = [d.strip() for d in date_range.split("to")]
        fb = calendar_service.freebusy().query(
            body={
                "timeMin": f"{start}T00:00:00Z",
                "timeMax": f"{end}T23:59:59Z",
                "items": [{"id": CALENDAR_ID}]
            }
        ).execute()

        busy = fb["calendars"][CALENDAR_ID]["busy"]
        if not busy:
            return f"All slots are free from {start} to {end}."
        busy_str = "; ".join(f"{b['start']} → {b['end']}" for b in busy)
        return f"Busy slots: {busy_str}"
    except Exception as e:
        logger.error(f"suggest_slots error: {e}", exc_info=True)
        return f"Error suggesting slots: {e}"

def book_appointment(input_str: str) -> str:
    if not calendar_service:
        return _no_service_msg()
    try:
        date_time, user_name = (
            [x.strip() for x in input_str.split(",", 1)]
            if "," in input_str else input_str.strip().split(" ", 1)
        )
        start_dt = datetime.fromisoformat(date_time.replace("Z", "+00:00"))
        end_dt = start_dt + timedelta(hours=1)
        event = {
            "summary": f"Appointment with {user_name}",
            "start": {"dateTime": start_dt.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": "UTC"},
        }
        created = calendar_service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
        return f"Booked {user_name} at {date_time}. Link: {created.get('htmlLink')}"
    except Exception as e:
        logger.error(f"book_appointment error: {e}", exc_info=True)
        return f"Error booking appointment: {e}"

# ────── LangChain Agent & Tools ──────
tools = [
    Tool(name="check_availability", func=check_availability, description="Check calendar availability for a specific date (YYYY-MM-DD UTC)."),
    Tool(name="suggest_slots", func=suggest_slots, description="Suggest free/busy slots: 'YYYY-MM-DD to YYYY-MM-DD'."),
    Tool(name="book_appointment", func=book_appointment, description="Book a 1-hour slot: 'YYYY-MM-DDTHH:MM:SS Name' or '...,Name'")
]

st_secrets = _get_streamlit_secrets()
GROQ_API_KEY = st_secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not configured!")

llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY, temperature=0.2)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",
    memory=memory,
    verbose=True
)

# ────── FastAPI Setup ──────
app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing.")
    try:
        response = agent.run(req.message)
        return ChatResponse(response=response)
    except Exception as e:
        logger.error("/chat endpoint error", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

# ────── Optional Routes ──────
class MsgPayload(BaseModel):
    msg_id: int
    msg_name: str

messages_list: Dict[int, MsgPayload] = {}

@app.get("/")
def root():
    return {"message": "Hello from the Calendar-Booking bot 👋"}

@app.get("/about")
def about():
    return {"message": "Conversational AI Appointment Booking Assistant"}

@app.post("/messages/")
def add_msg(payload: MsgPayload):
    messages_list[payload.msg_id] = payload
    return {"message": payload}

@app.get("/messages")
def list_msgs():
    return {"messages": messages_list}
