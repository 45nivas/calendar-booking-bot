# streamlit_app.py – Calendar‑booking Chatbot (Streamlit + LangChain + Google Calendar)
# Updated: 2025‑07‑03 (Fixed: Timezone display issue)

import streamlit as st
import logging
import importlib.util
import os
import json
from datetime import datetime, timedelta
from typing import Optional, Tuple

from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_groq.chat_models import ChatGroq

from google.oauth2 import service_account
from googleapiclient.discovery import build

from pytz import timezone
import dateparser

# ─────────────── Logging ───────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ────── Helpers ──────

def _get_streamlit_secrets():
    return getattr(st, "secrets", {}) if importlib.util.find_spec("streamlit") else {}


# ────── Load Google credentials ──────
def _load_google_credentials() -> Optional[service_account.Credentials]:
    scopes = ["https://www.googleapis.com/auth/calendar"]
    st_secrets = _get_streamlit_secrets()

    creds_dict = st_secrets.get("google_credentials")
    if creds_dict:
        try:
            logger.info("Loaded Google credentials from st.secrets")
            return service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
        except Exception as e:
            logger.warning(f"Could not read creds from st.secrets: {e}")

    env_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if env_json:
        try:
            logger.info("Loaded Google credentials from env var")
            return service_account.Credentials.from_service_account_info(json.loads(env_json), scopes=scopes)
        except Exception as e:
            logger.warning(f"Env‑var creds invalid: {e}")

    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
    if os.path.exists(path):
        try:
            logger.info(f"Loaded Google credentials from file {path}")
            return service_account.Credentials.from_service_account_file(path, scopes=scopes)
        except Exception as e:
            logger.warning(f"File creds invalid: {e}")

    return None

# ────── Calendar Setup ──────
CALENDAR_ID = "my-maps-project@potent-howl-456013-k9.iam.gserviceaccount.com"

credentials = _load_google_credentials()
if credentials:
    calendar_service = build("calendar", "v3", credentials=credentials)
    logger.info(f"Using calendar: {CALENDAR_ID}")
else:
    calendar_service = None
    logger.error("Google Calendar credentials not found. Calendar tools disabled.")

# ────── Utility functions ──────
def _no_service_msg() -> str:
    return "Google Calendar is not configured (missing credentials)."

def _parse_user_datetime(dt_str: str) -> datetime:
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        dt = dateparser.parse(
            dt_str,
            settings={"TIMEZONE": "Asia/Kolkata", "RETURN_AS_TIMEZONE_AWARE": False},
        )
        if dt is None:
            raise ValueError(
                "Could not parse date/time. Use e.g. '2025-07-04T17:00:00' or '4 July 2025 5pm'."
            )
        return dt

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
                orderBy="startTime",
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
        return "Booked slots on " + date + ": " + "; ".join(booked)
    except Exception as e:
        logger.error("check_availability error", exc_info=True)
        return f"Error checking availability: {e}"

def suggest_slots(date_range: str) -> str:
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
        logger.error("suggest_slots error", exc_info=True)
        return f"Error suggesting slots: {e}"

def _split_input(input_str: str) -> Tuple[str, str]:
    if "," in input_str:
        return [x.strip() for x in input_str.split(",", 1)]
    return input_str.strip().split(" ", 1)

def book_appointment(input_str: str) -> str:
    if not calendar_service:
        return _no_service_msg()
    try:
        date_time_str, user_name = _split_input(input_str)

        ist = timezone("Asia/Kolkata")
        user_dt = _parse_user_datetime(date_time_str)
        ist_dt = ist.localize(user_dt) if user_dt.tzinfo is None else user_dt.astimezone(ist)

        event = {
            "summary": f"Appointment with {user_name}",
            "start": {"dateTime": ist_dt.isoformat(), "timeZone": "Asia/Kolkata"},
            "end": {"dateTime": (ist_dt + timedelta(hours=1)).isoformat(), "timeZone": "Asia/Kolkata"},
        }

        logger.info(f"Creating event: {event}")
        created = calendar_service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
        event_link = created.get("htmlLink", "Link unavailable")
        return (
            f"Booked {user_name} at {ist_dt.strftime('%Y-%m-%d %I:%M %p')} IST. "
            f"Link: {event_link}"
        )
    except Exception as e:
        logger.error("book_appointment error", exc_info=True)
        return f"Error booking appointment: {e}"

# ────── LangChain Agent Setup ──────
TOOLS = [
    Tool(
        name="check_availability",
        func=check_availability,
        description="Check calendar availability for a specific date (YYYY-MM-DD UTC).",
    ),
    Tool(
        name="suggest_slots",
        func=suggest_slots,
        description="Suggest free/busy slots: 'YYYY-MM-DD to YYYY-MM-DD'.",
    ),
    Tool(
        name="book_appointment",
        func=book_appointment,
        description="Book a 1‑hour slot: 'YYYY-MM-DDTHH:MM:SS Name' or '...,Name'",
    ),
]

st_secrets = _get_streamlit_secrets()
GROQ_API_KEY = st_secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not configured!")

llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY, temperature=0.2)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=TOOLS,
    llm=llm,
    agent="chat-conversational-react-description",
    memory=memory,
    verbose=True,
)

# ────── Streamlit UI ──────
st.title("Calendar Booking Bot 🤖📅")
st.write("Ask me to check availability, suggest slots, or book an appointment!")

if "history" not in st.session_state:
    st.session_state.history = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = ""

def _on_send():
    user_input = st.session_state.pending_input
    if not user_input:
        return
    try:
        answer = agent.run(user_input)
    except Exception as e:
        answer = f"Error: {e}"
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", answer))
    st.session_state.pending_input = ""

st.text_input("Your message:", key="pending_input", on_change=_on_send)

for speaker, msg in st.session_state.history:
    st.markdown(f"**{speaker}:** {msg}")
