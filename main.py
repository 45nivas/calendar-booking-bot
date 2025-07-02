import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from datetime import datetime, timedelta

from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_groq.chat_models import ChatGroq

from google.oauth2 import service_account
from googleapiclient.discovery import build

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def share_calendar_with_user(service, calendar_id, user_email, role="owner"):
    """
    Share the Google Calendar with a user.
    role: 'owner' (full control) or 'writer' (edit access)
    """
    try:
        rule = {
            'scope': {
                'type': 'user',
                'value': user_email,
            },
            'role': role
        }
        created_rule = service.acl().insert(calendarId=calendar_id, body=rule).execute()
        logger.info(f"Shared calendar with {user_email}. Rule ID: {created_rule['id']}")
    except Exception as e:
        logger.error(f"Failed to share calendar with {user_email}: {e}")

# --- Google Calendar Setup ---
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")

try:
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    calendar_service = build("calendar", "v3", credentials=credentials)
    # --- Share calendar with a user (call this ONCE, not every run in production) ---
    share_calendar_with_user(calendar_service, CALENDAR_ID, "mattanivas37@gmail.com", role="owner")
except Exception as e:
    logger.error(f"Google Calendar setup failed: {e}")
    calendar_service = None

# --- Calendar Tool Functions ---
def check_availability(date: str) -> str:
    try:
        events_result = calendar_service.events().list(
            calendarId=CALENDAR_ID,
            timeMin=f"{date}T00:00:00Z",
            timeMax=f"{date}T23:59:59Z",
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        events = events_result.get("items", [])
        if not events:
            return f"No events found on {date}. All slots are available."
        else:
            booked = [f"{e['start'].get('dateTime', e['start'].get('date'))} - {e.get('summary', 'No title')}" for e in events]
            return f"Booked slots on {date}: " + "; ".join(booked)
    except Exception as e:
        logger.error(f"Error in check_availability: {e}")
        return f"Failed to check availability: {e}"

def suggest_slots(date_range: str) -> str:
    try:
        start, end = [d.strip() for d in date_range.split("to")]
        freebusy_query = {
            "timeMin": f"{start}T00:00:00Z",
            "timeMax": f"{end}T23:59:59Z",
            "items": [{"id": CALENDAR_ID}]
        }
        fb = calendar_service.freebusy().query(body=freebusy_query).execute()
        busy = fb["calendars"][CALENDAR_ID]["busy"]
        if not busy:
            return f"All slots are free from {start} to {end}."
        else:
            busy_str = "; ".join([f"{b['start']} to {b['end']}" for b in busy])
            return f"Busy slots: {busy_str}"
    except Exception as e:
        logger.error(f"Error in suggest_slots: {e}")
        return f"Failed to suggest slots: {e}"

def book_appointment(input_str: str) -> str:
    """
    Accepts input as '2025-07-06T14:00:00 John' or '2025-07-06T14:00:00,John'
    and splits into date_time and user_name.
    """
    try:
        # Try to split by space or comma
        if ',' in input_str:
            date_time, user_name = [x.strip() for x in input_str.split(',', 1)]
        else:
            date_time, user_name = input_str.strip().split(' ', 1)
        dt = datetime.fromisoformat(date_time.replace("Z", "+00:00"))
        end_time = (dt + timedelta(hours=1)).isoformat()
        event = {
            "summary": f"Appointment with {user_name}",
            "start": {"dateTime": dt.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": end_time, "timeZone": "UTC"},
            "attendees": [],
        }
        created = calendar_service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
        return f"Booked appointment for {user_name} at {date_time}. Event link: {created.get('htmlLink')}"
    except Exception as e:
        logger.error(f"Error in book_appointment: {e}")
        return f"Failed to book appointment: {e}"

# --- LangChain Tool Setup ---
tools = [
    Tool(
        name="check_availability",
        func=check_availability,
        description="Check calendar availability for a given date (format YYYY-MM-DD)."
    ),
    Tool(
        name="suggest_slots",
        func=suggest_slots,
        description="Suggest available slots (format: 'YYYY-MM-DD to YYYY-MM-DD')."
    ),
    Tool(
        name="book_appointment",
        func=book_appointment,
        description="Book an appointment. Input format: 'YYYY-MM-DDTHH:MM:SS Name' or 'YYYY-MM-DDTHH:MM:SS,Name'"
    )
]

# --- FastAPI ---
app = FastAPI()

# --- Groq LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not set in environment!")
    raise RuntimeError("Set your GROQ_API_KEY environment variable.")

llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY,
    temperature=0.2
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools,
    llm,
    agent="chat-conversational-react-description",
    memory=memory,
    verbose=True
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        user_message = request.message
        agent_response = agent.run(user_message)
        return ChatResponse(response=agent_response)
    except Exception as e:
        logger.error(f"Error in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

# --- Optional Message Routes ---
class MsgPayload(BaseModel):
    msg_id: int
    msg_name: str

messages_list: Dict[int, MsgPayload] = {}

@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hello"}

@app.get("/about")
def about() -> dict[str, str]:
    return {"message": "Conversational AI Appointment Booking Assistant."}

@app.post("/messages/")
def add_msg(payload: MsgPayload) -> dict[str, MsgPayload]:
    messages_list[payload.msg_id] = payload
    return {"message": payload}

@app.get("/messages")
def message_items() -> dict[str, Dict[int, MsgPayload]]:
    return {"messages": messages_list}
