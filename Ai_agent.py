"""
AI Menstrual Cycle Agent — FastAPI Backend (Hackathon-Ready)
============================================================

Single-file FastAPI app you can run immediately.

Features
- Users: lightweight auth via bearer tokens (signup/login)
- Cycle logging: start/end dates, auto-length calc, rolling averages
- Predictions: next period & ovulation, per-user
- Symptoms & mood logging (text/voice transcription handled client-side)
- AI chat: calls Gemini if GOOGLE_API_KEY is set, else graceful fallback
- Health risk flags: irregular cycles heuristics (non-diagnostic)
- Partner support mode: notification hook (console stub)
- Background tasks for notifications
- SQLite via SQLModel for speed; swap to Mongo later if you want

How to run
1) `pip install -r requirements.txt` (see bottom of file)
2) (optional) create a `.env` file with:
   GOOGLE_API_KEY=your_key_here
   SECRET_KEY=change_me
3) `uvicorn app:app --reload`

Try it in your browser:
- Docs: http://127.0.0.1:8000/docs

Security note
- This is hackathon-grade. For production, add proper rate limiting, JWT rotation, secrets management, audit logging, etc.
"""

from __future__ import annotations
import os
import secrets
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Body, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from sqlmodel import SQLModel, Field as ORMField, Session, create_engine, select, Relationship

# Optional: Gemini SDK (graceful import)
try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DB_URL = os.getenv("DB_URL", "sqlite:///./cycle_agent.db")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

engine = create_engine(DB_URL, echo=False)

security = HTTPBearer(auto_error=False)

app = FastAPI(title="AI Menstrual Cycle Agent — Backend", version="0.1.0")

# ----------------------------------------------------------------------------
# Database Models (SQLModel)
# ----------------------------------------------------------------------------
class UserToken(SQLModel, table=True):
    id: Optional[int] = ORMField(default=None, primary_key=True)
    user_id: int = ORMField(index=True)
    token: str = ORMField(index=True, unique=True)
    created_at: datetime = ORMField(default_factory=datetime.utcnow)

class User(SQLModel, table=True):
    id: Optional[int] = ORMField(default=None, primary_key=True)
    email: EmailStr = ORMField(index=True, unique=True)
    partner_contact: Optional[str] = None  # email or phone
    avg_cycle_length: Optional[int] = 28

    cycles: List["Cycle"] = Relationship(back_populates="user")
    symptoms: List["SymptomLog"] = Relationship(back_populates="user")

class Cycle(SQLModel, table=True):
    id: Optional[int] = ORMField(default=None, primary_key=True)
    user_id: int = ORMField(foreign_key="user.id")
    start_date: date
    end_date: Optional[date] = None

    user: Optional[User] = Relationship(back_populates="cycles")

class SymptomLog(SQLModel, table=True):
    id: Optional[int] = ORMField(default=None, primary_key=True)
    user_id: int = ORMField(foreign_key="user.id")
    day: date = ORMField(default_factory=lambda: datetime.utcnow().date())
    mood: Optional[str] = None
    symptoms: Optional[str] = None  # comma-separated for simplicity
    flow_intensity: Optional[str] = Field(default=None, description="light|medium|heavy")

    user: Optional[User] = Relationship(back_populates="symptoms")


# ----------------------------------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------------------------------
class SignupRequest(BaseModel):
    email: EmailStr
    partner_contact: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr

class AuthResponse(BaseModel):
    token: str

class CycleCreate(BaseModel):
    start_date: date
    end_date: Optional[date] = None

class SymptomCreate(BaseModel):
    day: Optional[date] = None
    mood: Optional[str] = None
    symptoms: Optional[List[str]] = None
    flow_intensity: Optional[str] = Field(default=None, regex=r"^(light|medium|heavy)$")

class PredictionResponse(BaseModel):
    next_period: date
    ovulation: date
    cycle_day_today: Optional[int] = None

class RiskFlag(BaseModel):
    type: str
    message: str

class InsightResponse(BaseModel):
    avg_cycle_length: Optional[int]
    last_cycle_length: Optional[int]
    risk_flags: List[RiskFlag]

class ChatRequest(BaseModel):
    cycle_day: Optional[int] = None
    mood: Optional[str] = None
    symptoms: Optional[List[str]] = None
    question: str

class ChatResponse(BaseModel):
    answer: str
    used_gemini: bool


# ----------------------------------------------------------------------------
# DB Setup
# ----------------------------------------------------------------------------
SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


# ----------------------------------------------------------------------------
# Auth Helpers (naive tokens, hackathon-friendly)
# ----------------------------------------------------------------------------

def create_token(session: Session, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    session.add(UserToken(user_id=user_id, token=token))
    session.commit()
    return token


def user_from_token(session: Session, token: str) -> User:
    tok = session.exec(select(UserToken).where(UserToken.token == token)).first()
    if not tok:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = session.get(User, tok.user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
    session: Session = Depends(get_session),
) -> User:
    if creds is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    return user_from_token(session, creds.credentials)


# ----------------------------------------------------------------------------
# Utility Logic
# ----------------------------------------------------------------------------

def cycle_lengths(cycles: List[Cycle]) -> List[int]:
    lengths = []
    # sort by start_date
    cs = sorted([c for c in cycles if c.start_date], key=lambda c: c.start_date)
    for i in range(1, len(cs)):
        delta = (cs[i].start_date - cs[i-1].start_date).days
        if 15 <= delta <= 120:  # sanity bounds
            lengths.append(delta)
    return lengths


def rolling_avg_length(lengths: List[int]) -> Optional[int]:
    if not lengths:
        return None
    return int(round(sum(lengths) / len(lengths)))


def predict_cycle(last_start: date, avg_cycle: int = 28) -> tuple[date, date]:
    next_period = last_start + timedelta(days=avg_cycle)
    ovulation = last_start + timedelta(days=avg_cycle - 14)
    return next_period, ovulation


def cycle_day_today(cycles: List[Cycle]) -> Optional[int]:
    if not cycles:
        return None
    last = max(cycles, key=lambda c: c.start_date)
    return (datetime.utcnow().date() - last.start_date).days + 1


def irregularity_flags(lengths: List[int], avg_len: Optional[int]) -> List[RiskFlag]:
    flags: List[RiskFlag] = []
    if not lengths:
        return flags
    for L in lengths[-3:]:
        if L < 21 or L > 40:
            flags.append(RiskFlag(type="irregular_length", message=f"Cycle length {L} days is outside 21–40 day range. Consider monitoring; consult a clinician if persistent."))
    if avg_len is not None:
        for L in lengths[-3:]:
            if abs(L - avg_len) > 7:
                flags.append(RiskFlag(type="high_variability", message=f"Cycle length {L} deviates >7 days from your average ({avg_len})."))
    return flags


def notify_partner_stub(partner_contact: Optional[str], message: str) -> None:
    if not partner_contact:
        return
    # In a real app, integrate Twilio/Sendgrid/WhatsApp here.
    print(f"[Partner Notification] to {partner_contact}: {message}")


# ----------------------------------------------------------------------------
# Routes: Auth
# ----------------------------------------------------------------------------
@app.post("/signup", response_model=AuthResponse)
def signup(payload: SignupRequest, session: Session = Depends(get_session)):
    existing = session.exec(select(User).where(User.email == payload.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=payload.email, partner_contact=payload.partner_contact)
    session.add(user)
    session.commit()
    session.refresh(user)
    token = create_token(session, user.id)
    return AuthResponse(token=token)


@app.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == payload.email)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found. Please sign up.")
    token = create_token(session, user.id)
    return AuthResponse(token=token)


@app.get("/me")
def me(user: User = Depends(current_user)):
    return {"email": str(user.email), "partner_contact": user.partner_contact, "avg_cycle_length": user.avg_cycle_length}


# ----------------------------------------------------------------------------
# Routes: Cycles & Predictions
# ----------------------------------------------------------------------------
@app.post("/cycles")
def add_cycle(
    data: CycleCreate,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
    background: BackgroundTasks = None,
):
    c = Cycle(user_id=user.id, start_date=data.start_date, end_date=data.end_date)
    session.add(c)
    session.commit()

    # recompute averages
    user_cycles = session.exec(select(Cycle).where(Cycle.user_id == user.id)).all()
    lengths = cycle_lengths(user_cycles)
    avg_len = rolling_avg_length(lengths)
    if avg_len:
        user.avg_cycle_length = avg_len
        session.add(user)
        session.commit()

    # optional partner tip for Day 1-3
    today_day = cycle_day_today(user_cycles)
    if today_day and 1 <= today_day <= 3:
        background.add_task(
            notify_partner_stub,
            user.partner_contact,
            "She might be experiencing cramps or low energy today. Be kind and supportive ❤️",
        )

    return {"ok": True, "avg_cycle_length": user.avg_cycle_length}


@app.get("/predictions", response_model=PredictionResponse)
def get_predictions(user: User = Depends(current_user), session: Session = Depends(get_session)):
    user_cycles = session.exec(select(Cycle).where(Cycle.user_id == user.id)).all()
    if not user_cycles:
        raise HTTPException(status_code=400, detail="No cycles logged yet")

    last_start = max(user_cycles, key=lambda c: c.start_date).start_date
    avg_len = user.avg_cycle_length or 28
    next_period, ovulation = predict_cycle(last_start, avg_len)
    day_today = cycle_day_today(user_cycles)
    return PredictionResponse(next_period=next_period, ovulation=ovulation, cycle_day_today=day_today)


# ----------------------------------------------------------------------------
# Routes: Symptoms & Insights
# ----------------------------------------------------------------------------
@app.post("/symptoms")
def log_symptom(data: SymptomCreate, user: User = Depends(current_user), session: Session = Depends(get_session)):
    s = SymptomLog(
        user_id=user.id,
        day=data.day or datetime.utcnow().date(),
        mood=data.mood,
        symptoms=",".join(data.symptoms) if data.symptoms else None,
        flow_intensity=data.flow_intensity,
    )
    session.add(s)
    session.commit()
    return {"ok": True}


@app.get("/insights", response_model=InsightResponse)
def insights(user: User = Depends(current_user), session: Session = Depends(get_session)):
    user_cycles = session.exec(select(Cycle).where(Cycle.user_id == user.id)).all()
    lengths = cycle_lengths(user_cycles)
    avg_len = rolling_avg_length(lengths) or user.avg_cycle_length
    last_len = lengths[-1] if lengths else None
    flags = irregularity_flags(lengths, avg_len)
    return InsightResponse(avg_cycle_length=avg_len, last_cycle_length=last_len, risk_flags=flags)


# ----------------------------------------------------------------------------
# Routes: AI Chat (Gemini -> fallback)
# ----------------------------------------------------------------------------

def call_gemini(question: str, cycle_day: Optional[int], mood: Optional[str], symptoms: Optional[List[str]]) -> str:
    # Build the structured prompt
    prompt = (
        "You are a compassionate, evidence-informed women's health assistant.\n"
        "Be clear, supportive, and non-judgmental.\n"
        "Avoid diagnosis; suggest when to seek clinical care.\n\n"
        f"Cycle day: {cycle_day if cycle_day is not None else 'unknown'}\n"
        f"Mood: {mood or 'unspecified'}\n"
        f"Symptoms: {', '.join(symptoms) if symptoms else 'unspecified'}\n\n"
        f"User question: {question}\n\n"
        "In 4-7 sentences: 1) explain what's likely happening hormonally this phase, "
        "2) offer 2-3 tailored self-care tips (sleep, hydration, nutrition, movement), "
        "3) add 1 red-flag that warrants seeing a clinician if present."
    )

    if GOOGLE_API_KEY and genai:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            return resp.text.strip() if hasattr(resp, "text") else str(resp)
        except Exception as e:  # pragma: no cover
            return f"(Gemini error fallback) Here's a supportive response based on your inputs. Reason: {e}"
    else:
        # Fallback: rule-of-thumb message
        base = "It’s common for mood and energy to shift across the cycle. "
        tips = (
            "Try steady meals with protein and complex carbs, keep water nearby, and aim for gentle movement like a walk. "
            "If cramps or low mood feel intense or last >2 weeks, consider checking in with a clinician."
        )
        return base + tips


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, user: User = Depends(current_user)):
    answer = call_gemini(
        question=payload.question,
        cycle_day=payload.cycle_day,
        mood=payload.mood,
        symptoms=payload.symptoms,
    )
    return ChatResponse(answer=answer, used_gemini=bool(GOOGLE_API_KEY and genai))


# ----------------------------------------------------------------------------
# Partner Support Endpoint (optional)
# ----------------------------------------------------------------------------
class PartnerUpdate(BaseModel):
    partner_contact: Optional[str]


@app.post("/partner")
def set_partner(data: PartnerUpdate, user: User = Depends(current_user), session: Session = Depends(get_session)):
    user.partner_contact = data.partner_contact
    session.add(user)
    session.commit()
    return {"ok": True, "partner_contact": user.partner_contact}


# ----------------------------------------------------------------------------
# Dev Convenience: seed endpoint
# ----------------------------------------------------------------------------
@app.post("/dev/seed")
def dev_seed(session: Session = Depends(get_session)):
    # creates a sample user and a couple of cycles
    email = EmailStr("demo@example.com")
    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        user = User(email=email, partner_contact="partner@example.com")
        session.add(user)
        session.commit()
        session.refresh(user)
        create_token(session, user.id)

    # add sample cycles
    existing = session.exec(select(Cycle).where(Cycle.user_id == user.id)).all()
    if not existing:
        session.add(Cycle(user_id=user.id, start_date=date(2025, 1, 1), end_date=date(2025, 1, 5)))
        session.add(Cycle(user_id=user.id, start_date=date(2025, 1, 29), end_date=date(2025, 2, 2)))
        session.add(Cycle(user_id=user.id, start_date=date(2025, 2, 26), end_date=date(2025, 3, 2)))
        session.commit()

    return {"ok": True}


# ----------------------------------------------------------------------------
# Minimal requirements.txt you can copy-paste to a file
# ----------------------------------------------------------------------------
REQUIREMENTS_TXT = """
fastapi==0.115.5
uvicorn[standard]==0.32.0
sqlmodel==0.0.22
pydantic==2.9.2
python-dotenv==1.0.1
google-generativeai==0.8.3
"""


@app.get("/requirements.txt")
def requirements_text():
    return REQUIREMENTS_TXT
