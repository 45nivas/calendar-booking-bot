import streamlit as st
import logging
import importlib.util
import os
import json
from datetime import datetime
from typing import Optional, Tuple

from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_groq.chat_models import ChatGroq

from google.oauth2 import service_account
from googleapiclient.discovery import build

from pytz import timezone
import dateparser
import re

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

def extract_datetime_range(input_str: str) -> tuple:
    date_match = re.search(r'on\s+(.*?)(?:\sat|\sfrom|\sbetween)', input_str, re.IGNORECASE)
    if not date_match:
        raise ValueError("Could not find a date in the input.")
    date_str = date_match.group(1).strip()

    time_match = re.search(r'at\s+([0-9:apm\s]+)\s*(?:to|-)\s*([0-9:apm\s]+)', input_str, re.IGNORECASE)
    if not time_match:
        raise ValueError("Could not find a time range in the input.")
    start_time_str, end_time_str = time_match.groups()

    person_match = re.search(r'with\s+([\w\s]+?)(?=\s+on|\s+at)', input_str, re.IGNORECASE)
    person = person_match.group(1).strip() if person_match else "Guest"

    start_dt = dateparser.parse(f"{date_str} {start_time_str}", settings={"TIMEZONE": "Asia/Kolkata"})
    end_dt = dateparser.parse(f"{date_str} {end_time_str}", settings={"TIMEZONE": "Asia/Kolkata"})

    if not start_dt or not end_dt:
        raise ValueError("Could not parse start or end datetime.")

    return start_dt, end_dt, person

def book_appointment(input_str: str) -> str:
    if not calendar_service:
        return _no_service_msg()
    try:
        start_dt, end_dt, user_name = extract_datetime_range(input_str)

        ist = timezone("Asia/Kolkata")
        ist_start = ist.localize(start_dt) if start_dt.tzinfo is None else start_dt.astimezone(ist)
        ist_end = ist.localize(end_dt) if end_dt.tzinfo is None else end_dt.astimezone(ist)

        if ist_start >= ist_end:
            raise ValueError("The end time must be after the start time.")

        event = {
            "summary": f"Appointment with {user_name}",
            "start": {"dateTime": ist_start.isoformat(), "timeZone": "Asia/Kolkata"},
            "end": {"dateTime": ist_end.isoformat(), "timeZone": "Asia/Kolkata"},
        }

        created = calendar_service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
        event_link = created.get("htmlLink", "Link unavailable")

        return (
            f"✅ Booked appointment with **{user_name}** on **{ist_start.strftime('%d %B %Y')}** "
            f"from **{ist_start.strftime('%I:%M %p')}** to **{ist_end.strftime('%I:%M %p')}** IST.\n"
            f"[View event]({event_link})"
        )
    except Exception as e:
        logger.error("book_appointment error", exc_info=True)
        return f"❌ Failed to book appointment: {e}"

def check_availability(date: str) -> str:
    if not calendar_service:
        return _no_service_msg()
    try:
        events = calendar_service.events().list(
            calendarId=CALENDAR_ID,
            timeMin=f"{date}T00:00:00Z",
            timeMax=f"{date}T23:59:59Z",
            singleEvents=True,
            orderBy="startTime",
        ).execute().get("items", [])

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
        fb = calendar_service.freebusy().query(
            body={
                "timeMin": f"{start}T00:00:00Z",
                "timeMax": f"{end}T23:59:59Z",
                "items": [{"id": CALENDAR_ID}],
            }
        ).execute()
        busy = fb["calendars"][CALENDAR_ID]["busy"]
        if not busy:
            return f"All slots are free from {start} to {end}."
        busy_str = "; ".join(f"{b['start']} → {b['end']}" for b in busy)
        return f"Busy slots: {busy_str}"
    except Exception as e:
        logger.error("suggest_slots error", exc_info=True)
        return f"Error suggesting slots: {e}"

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
        description="Book an appointment using natural language like 'book with Amma on July 11th, 2025 at 4pm to 7pm'",
    ),
]

# Load LLM
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
st.title("🧠📅 AI Calendar Booking Bot")
st.write("Chat with me to check availability or book appointments with custom timings!")

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