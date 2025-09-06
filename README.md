[![🧠 Live Calendar Booking App](https://img.shields.io/badge/🚀%20Live%20App-Click%20Here-success?style=for-the-badge)](https://calendar-booking-bot-ty3fdmvpd6smrneuosax8y.streamlit.app/)

# 🧠📅 AI Calendar Booking Bot

This is a conversational AI bot built with **LangChain**, **Streamlit**, and **Google Calendar API** that allows users to **book appointments** using natural language, like:

```
book appointment with Amma on July 11th, 2025 at 4pm to 7pm
```

## ✅ Features
- Natural language booking using Groq LLaMA 3 (70B)
- Date/time parsing with `dateparser`
- Google Calendar integration with start and end timea
- Check availability or suggest free/busy slots
- Hosted frontend with Streamlit

## 🧠 How It Works
- **LangChain tools** manage:
  - `book_appointment`
  - `check_availability`
  - `suggest_slots`
- **LangChain Agent** invokes tools using Groq API
- **Google Calendar API** inserts and retrieves events

## 🛠️ Technologies
- Python
- Streamlit
- LangChain + Groq API (LLaMA 3)
- Google Calendar API
- Dateparser


## 🔑 Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/ai-calendar-bot
cd ai-calendar-bot
```

### 2. Add `.streamlit/secrets.toml`
```toml
GROQ_API_KEY = "your-groq-api-key"

[google_credentials]
type = "service_account"
project_id = "your-project-id"
private_key_id = "xxx"
private_key = "-----BEGIN PRIVATE KEY-----\nXXX\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "..."
auth_uri = "..."
token_uri = "..."
auth_provider_x509_cert_url = "..."
client_x509_cert_url = "..."
```

### 3. Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 4. Deploy (Railway/Render/etc.)
Use `requirements.txt` and `streamlit_app.py` for deployment.

---
