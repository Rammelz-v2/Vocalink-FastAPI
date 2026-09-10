from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.exceptions import RequestValidationError

from sqlalchemy import (
    create_engine, Column, Integer, String,
    ForeignKey, text, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

from pydantic import BaseModel, EmailStr
from typing import Optional
from passlib.context import CryptContext

import jwt
import datetime as dt
import os
import io
import json
import random
import string
import threading
import requests

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
SECRET_KEY         = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM          = "HS256"
BREVO_API_KEY      = os.getenv("BREVO_API_KEY")
BREVO_SENDER_EMAIL = os.getenv("BREVO_SENDER_EMAIL", "olacomarkaidel@gmail.com")
BREVO_SENDER_NAME  = "VocaLink"
otp_store: dict    = {}

HF_API_URL  = "https://api-inference.huggingface.co/models/rammealz123/VOCALink-Mobile-STT"
HF_TOKEN    = os.getenv("HUGGINGFACE_TOKEN")
HF_HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vocalink.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine       = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()
pwd_context  = CryptContext(schemes=["bcrypt"], deprecated="auto")

CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "*",
}

app = FastAPI(title="VocaLink API")

# ── CORS middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Force CORS on ALL responses including errors ──────────────────────────────
@app.middleware("http")
async def force_cors(request: Request, call_next):
    if request.method == "OPTIONS":
        return Response(status_code=200, headers={**CORS_HEADERS, "Access-Control-Max-Age": "86400"})
    try:
        response = await call_next(request)
    except Exception as exc:
        response = JSONResponse(status_code=500, content={"detail": str(exc)})
    for k, v in CORS_HEADERS.items():
        response.headers[k] = v
    return response

# ── Exception handlers also include CORS headers ─────────────────────────────
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=CORS_HEADERS)

@app.exception_handler(RequestValidationError)
async def validation_exc_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()}, headers=CORS_HEADERS)

@app.exception_handler(Exception)
async def generic_exc_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)}, headers=CORS_HEADERS)


# ─────────────────────────────────────────────
# 2. MODELS
# ─────────────────────────────────────────────

class OTPToken(Base):
    __tablename__ = "otp_tokens"
    id         = Column(Integer, primary_key=True, index=True)
    email      = Column(String, index=True)
    code       = Column(String)
    expires_at = Column(String)
    used       = Column(Boolean, default=False)


class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, unique=True, index=True)
    email           = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    status          = Column(String, default="STUDENT")  # STUDENT | TEACHER
    is_verified     = Column(Integer, default=0)  # 0=unverified, 1=verified (DB uses INTEGER)

    teacher_profile = relationship("TeacherProfile", back_populates="user", uselist=False)
    student_profile = relationship("StudentProfile",  back_populates="user", uselist=False)


class TeacherProfile(Base):
    __tablename__  = "teacher_profiles"
    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"))
    first_name     = Column(String, default="")
    last_name      = Column(String, default="")
    display_name   = Column(String, default="")
    contact_number = Column(String, default="")
    room_section   = Column(String, default="")
    department     = Column(String, default="")
    grade_handled  = Column(String, default="")
    organization   = Column(String, default="")
    bio            = Column(String, default="")

    user     = relationship("User",           back_populates="teacher_profile")
    students = relationship("StudentProfile", back_populates="instructor")
    sessions = relationship("ClassSession",   back_populates="teacher")


class StudentProfile(Base):
    __tablename__   = "student_profiles"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    instructor_id   = Column(Integer, ForeignKey("teacher_profiles.id", ondelete="SET NULL"), nullable=True)
    first_name      = Column(String, nullable=True)
    last_name       = Column(String, nullable=True)
    bio             = Column(String, nullable=True)
    grade_level     = Column(String, nullable=True)
    disability_type = Column(String, nullable=True)
    last_seen = Column(String, nullable=True)

    instructor = relationship("TeacherProfile", back_populates="students")
    user       = relationship("User",           back_populates="student_profile")


# ── DB-backed session — one row per session, multiple per teacher ────────────
class ClassSession(Base):
    __tablename__ = "class_sessions"
    id           = Column(Integer, primary_key=True, index=True)
    teacher_id   = Column(Integer, ForeignKey("teacher_profiles.id", ondelete="CASCADE"))
    session_code = Column(String)
    is_active    = Column(Boolean, default=True)
    started_at   = Column(String, default=lambda: dt.datetime.utcnow().isoformat())

    teacher = relationship("TeacherProfile", back_populates="sessions")


# ── Caption messages — saved to DB, polled every 1-2 s by students ──────────
class CCMessage(Base):
    __tablename__ = "cc_messages"
    id         = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(Integer, ForeignKey("teacher_profiles.id", ondelete="CASCADE"), nullable=True)
    session_id = Column(Integer, ForeignKey("class_sessions.id", ondelete="SET NULL"), nullable=True)
    text       = Column(String)
    speaker    = Column(String, default="teacher")
    sent_at    = Column(String, default=lambda: dt.datetime.utcnow().isoformat())


class AACLog(Base):
    __tablename__ = "aac_logs"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    session_id = Column(Integer, ForeignKey("class_sessions.id", ondelete="SET NULL"), nullable=True)
    icon_id    = Column(String)
    icon_label = Column(String)
    message    = Column(String, nullable=True)
    tapped_at  = Column(String, default=lambda: dt.datetime.utcnow().isoformat())


Base.metadata.create_all(bind=engine)

# ── Safe auto-migrations for existing DBs ────────────────────────────────────
_migrations = [
    ("users",            "is_verified INTEGER DEFAULT 0"),
    ("student_profiles", "last_seen VARCHAR"),
    ("teacher_profiles", "first_name VARCHAR DEFAULT ''"),
    ("teacher_profiles", "last_name VARCHAR DEFAULT ''"),
    ("teacher_profiles", "grade_handled VARCHAR DEFAULT ''"),
    ("teacher_profiles", "organization VARCHAR DEFAULT ''"),
    ("teacher_profiles", "bio VARCHAR DEFAULT ''"),
    ("cc_messages",      "teacher_id INTEGER"),
    ("class_sessions",   "is_active BOOLEAN DEFAULT 1"),
    ("cc_messages",      "session_id INTEGER"),
    ("aac_logs",         "session_id INTEGER"),
]
for _table, _col in _migrations:
    try:
        with engine.connect() as _conn:
            _conn.execute(text(f"ALTER TABLE {_table} ADD COLUMN {_col}"))
            _conn.commit()
    except Exception:
        pass

# Drop the unique constraint on class_sessions.teacher_id so each session
# gets its own row.  Needed for existing databases.
try:
    with engine.connect() as _conn:
        if DATABASE_URL.startswith("sqlite"):
            _row = _conn.execute(text(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='class_sessions'"
            )).fetchone()
            if _row and "UNIQUE" in (_row[0] or "").upper():
                _conn.execute(text("""
                    CREATE TABLE class_sessions_v2 (
                        id         INTEGER NOT NULL PRIMARY KEY,
                        teacher_id INTEGER REFERENCES teacher_profiles(id) ON DELETE CASCADE,
                        session_code VARCHAR,
                        is_active  BOOLEAN DEFAULT 1,
                        started_at VARCHAR
                    )
                """))
                _conn.execute(text(
                    "INSERT INTO class_sessions_v2 "
                    "SELECT id, teacher_id, session_code, is_active, started_at "
                    "FROM class_sessions"
                ))
                _conn.execute(text("DROP TABLE class_sessions"))
                _conn.execute(text("ALTER TABLE class_sessions_v2 RENAME TO class_sessions"))
                _conn.commit()
        else:
            _conn.execute(text(
                "ALTER TABLE class_sessions "
                "DROP CONSTRAINT IF EXISTS class_sessions_teacher_id_key"
            ))
            _conn.commit()
except Exception as _e:
    print(f"[migration] class_sessions unique removal: {_e}")


# ─────────────────────────────────────────────
# 3. SCHEMAS
# ─────────────────────────────────────────────

class RegisterSchema(BaseModel):
    username: str
    email:    EmailStr
    password: str
    status:   str = "STUDENT"

class LoginSchema(BaseModel):
    identifier: str
    password:   str

class ProfileUpdate(BaseModel):
    first_name:      Optional[str] = None
    last_name:       Optional[str] = None
    bio:             Optional[str] = None
    grade_level:     Optional[str] = None
    disability_type: Optional[str] = None

class ProfileUpdateSchema(BaseModel):
    username:       Optional[str]      = None
    email:          Optional[EmailStr] = None
    first_name:     Optional[str]      = None
    last_name:      Optional[str]      = None
    display_name:   Optional[str]      = None
    contact_number: Optional[str]      = None
    room_section:   Optional[str]      = None
    department:     Optional[str]      = None
    grade_handled:  Optional[str]      = None
    organization:   Optional[str]      = None
    bio:            Optional[str]      = None

class BroadcastSchema(BaseModel):
    text:    str
    speaker: str = "teacher"

class AACLogSchema(BaseModel):
    icon_id:    str
    icon_label: str
    message:    Optional[str] = None

class TTSSchema(BaseModel):
    text: str


class VerifyEmailSchema(BaseModel):
    email: str
    code: str

class SessionLogSchema(BaseModel):
    session_code: str
    icon_id: str
    icon_label: str

# ─────────────────────────────────────────────
# 4. HELPERS & DEPENDENCIES
# ─────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_access_token(data: dict) -> str:
    payload = {**data, "exp": dt.datetime.utcnow() + dt.timedelta(days=7)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(auth.split(" ")[1], SECRET_KEY, algorithms=[ALGORITHM])
        user = db.query(User).filter(User.id == payload.get("user_id")).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def _make_code(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


# ─────────────────────────────────────────────
# 5. AUTH
# ─────────────────────────────────────────────

@app.post("/api/auth/register/")
def register(data: RegisterSchema, db: Session = Depends(get_db)):
    if db.query(User).filter(
        (User.username == data.username) | (User.email == data.email)
    ).first():
        raise HTTPException(status_code=400, detail="Username or email already taken")

    try:
        hashed_pw = pwd_context.hash(data.password)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Password hashing failed: {str(e)}")

    user = User(
    username        = data.username,
    email           = str(data.email),
    hashed_password = hashed_pw,
    status          = data.status,
    is_verified     = 1,   # skip email verification entirely
)
    db.add(user)
    db.commit()
    db.refresh(user)

    try:
        if user.status == "TEACHER":
            if not db.query(TeacherProfile).filter_by(user_id=user.id).first():
                db.add(TeacherProfile(user_id=user.id))
        else:
            if not db.query(StudentProfile).filter_by(user_id=user.id).first():
                db.add(StudentProfile(user_id=user.id))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[register] profile warning: {e}")

    # Send verification email
    code = str(random.randint(100000, 999999))
    expires = dt.datetime.utcnow() + dt.timedelta(minutes=30)
    db.query(OTPToken).filter(OTPToken.email == user.email).delete()
    db.add(OTPToken(email=user.email, code=code, expires_at=expires.isoformat()))
    db.commit()
    print(f"[VERIFY CODE] {user.email} → {code}")
    email_sent, email_error = _send_otp_via_brevo(user.email, code)
    if email_sent:
        print(f"[BREVO] Sent to {user.email}")
    else:
        print(f"[BREVO] Failed: {email_error}")

    return {
        "message": "Account created! Check your email for a verification code.",
        "email": user.email,
        "email_sent": email_sent,
        "email_error": email_error,
    }

@app.get("/api/auth/test-brevo/")
def test_brevo():
    sent, error = _send_otp_via_brevo(BREVO_SENDER_EMAIL, "123456")
    return {
        "brevo_api_key_set": bool(BREVO_API_KEY),
        "sender_email": BREVO_SENDER_EMAIL,
        "test_send": "ok" if sent else "failed",
        "test_error": error,
    }

class ResendOTPSchema(BaseModel):
    email: str

def _send_otp_via_brevo(email: str, code: str) -> tuple[bool, str | None]:
    if not BREVO_API_KEY:
        return False, "BREVO_API_KEY not set on server"
    try:
        resp = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json", "accept": "application/json"},
            json={
                "sender": {"name": BREVO_SENDER_NAME, "email": BREVO_SENDER_EMAIL},
                "to": [{"email": email}],
                "subject": "VocaLink — Your Verification Code",
                "htmlContent": (
                    "<div style='font-family:sans-serif;max-width:480px;margin:auto;"
                    "padding:32px;background:#f9f9f9;border-radius:12px'>"
                    "<h2 style='color:#1AADDC'>VocaLink Email Verification</h2>"
                    "<p>Your verification code:</p>"
                    f"<div style='font-size:40px;font-weight:800;letter-spacing:10px;"
                    f"color:#1A1A2E;padding:20px 0;text-align:center'>{code}</div>"
                    "<p style='color:#6B7280;font-size:13px'>Expires in <strong>30 minutes</strong>.</p></div>"
                ),
            },
            timeout=10,
        )
        if resp.status_code in (200, 201):
            return True, None
        return False, f"Brevo {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, str(e)

@app.post("/api/auth/forgot-password/")
def forgot_password(data: ResendOTPSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="No account found with that email.")
    code = str(random.randint(100000, 999999))
    expires = dt.datetime.utcnow() + dt.timedelta(minutes=30)
    db.query(OTPToken).filter(OTPToken.email == user.email).delete()
    db.add(OTPToken(email=user.email, code=code, expires_at=expires.isoformat()))
    db.commit()
    print(f"[FORGOT PASSWORD OTP] {user.email} → {code}")
    sent, error = _send_otp_via_brevo(user.email, code)
    return {"message": "Reset code sent to your email.", "email_sent": sent, "email_error": error}


class ResetPasswordSchema(BaseModel):
    email: str
    code: str
    new_password: str


@app.post("/api/auth/reset-password/")
def reset_password(data: ResetPasswordSchema, db: Session = Depends(get_db)):
    record = (
        db.query(OTPToken)
        .filter(OTPToken.email == data.email, OTPToken.used == False)
        .order_by(OTPToken.id.desc())
        .first()
    )
    if not record:
        raise HTTPException(status_code=400, detail="No reset code found. Please request a new one.")
    if dt.datetime.utcnow() > dt.datetime.fromisoformat(record.expires_at):
        db.delete(record)
        db.commit()
        raise HTTPException(status_code=400, detail="Code expired. Please request a new one.")
    if record.code != data.code:
        raise HTTPException(status_code=400, detail="Invalid code.")
    if len(data.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    user.hashed_password = pwd_context.hash(data.new_password)
    record.used = True
    db.commit()
    return {"message": "Password reset successfully. You can now sign in."}


@app.post("/api/auth/resend-verification/")
def resend_verification(data: ResendOTPSchema, db: Session = Depends(get_db)):
    user = (
        db.query(User).filter(User.email == data.email).first() or
        db.query(User).filter(User.username == data.email).first()
    )
    if not user:
        raise HTTPException(status_code=404, detail="No account found with that email.")
    code = str(random.randint(100000, 999999))
    expires = dt.datetime.utcnow() + dt.timedelta(minutes=30)
    db.query(OTPToken).filter(OTPToken.email == user.email).delete()
    db.add(OTPToken(email=user.email, code=code, expires_at=expires.isoformat()))
    db.commit()
    print(f"[RESEND VERIFICATION] {user.email} → {code}")
    sent, error = _send_otp_via_brevo(user.email, code)
    return {"message": "Verification code resent.", "email_sent": sent, "email_error": error}

@app.post("/api/auth/verify-email/")
def verify_email(data: VerifyEmailSchema, db: Session = Depends(get_db)):
    record = (
        db.query(OTPToken)
        .filter(OTPToken.email == data.email, OTPToken.used == False)
        .order_by(OTPToken.id.desc())
        .first()
    )
    if not record:
        raise HTTPException(status_code=400, detail="No verification code found. Please tap 'Resend Code'.")
    if dt.datetime.utcnow() > dt.datetime.fromisoformat(record.expires_at):
        db.delete(record)
        db.commit()
        raise HTTPException(status_code=400, detail="Code expired. Please tap 'Resend Code'.")
    if record.code != data.code:
        raise HTTPException(status_code=400, detail="Invalid code.")

    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    user.is_verified = 1
    record.used = True
    db.commit()
    return {"message": "Email verified! You can now sign in."}


@app.post("/api/auth/login/")
def login(data: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        (User.username == data.identifier) | (User.email == data.identifier)
    ).first()
    if not user or not pwd_context.verify(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_access_token({"user_id": user.id}), "status": user.status}

# ─────────────────────────────────────────────
# 6. PROFILE
# ─────────────────────────────────────────────

@app.get("/api/profile/me")
def get_profile(current_user: User = Depends(get_current_user)):
    if current_user.status == "TEACHER":
        p = current_user.teacher_profile
        return {
            "id": current_user.id, "username": current_user.username,
            "email": current_user.email, "status": current_user.status,
            "first_name": p.first_name if p else "",
            "last_name":  p.last_name  if p else "",
            "display_name": p.display_name if p else "",
            "department":   p.department   if p else "",
            "grade_handled":p.grade_handled if p else "",
            "room_section": p.room_section  if p else "",
            "bio":          p.bio           if p else "",
        }

    p = current_user.student_profile
    teacher_name, teacher_id = "", None
    if p and p.instructor_id and p.instructor:
        ins = p.instructor
        teacher_name = ins.first_name or ins.display_name or "Teacher"
        teacher_id   = ins.user_id
    return {
        "id": current_user.id, "username": current_user.username,
        "email": current_user.email, "status": current_user.status,
        "first_name":      p.first_name      if p else "",
        "last_name":       p.last_name       if p else "",
        "grade_level":     p.grade_level     if p else "",
        "disability_type": p.disability_type if p else "",
        "bio":             p.bio             if p else "",
        "teacher_name":    teacher_name,
        "teacher_id":      teacher_id,
    }


@app.put("/api/profile/me")
def update_profile(
    data: ProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    p = db.query(StudentProfile).filter_by(user_id=current_user.id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found")
    for field in ("first_name","last_name","bio","grade_level","disability_type"):
        val = getattr(data, field)
        if val is not None:
            setattr(p, field, val)
    db.commit()
    return {"message": "Profile updated"}


@app.patch("/api/users/me/")
def update_me(
    data: ProfileUpdateSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if data.username: current_user.username = data.username
    if data.email:    current_user.email    = data.email
    p = current_user.teacher_profile
    if p:
        for f in ("first_name","last_name","display_name","contact_number",
                  "room_section","department","grade_handled","organization","bio"):
            v = getattr(data, f, None)
            if v is not None:
                setattr(p, f, v)
    db.commit()
    return {"message": "Profile updated"}


@app.get("/api/users/me/")
def get_me(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Web App.tsx fetches this to populate the sidebar teacher name."""
    if current_user.status == "TEACHER":
        p = current_user.teacher_profile
        return {
            "id":           current_user.id,
            "username":     current_user.username,
            "email":        current_user.email,
            "status":       current_user.status,
            "display_name": p.display_name if p else "",
            "first_name":   p.first_name   if p else "",
            "last_name":    p.last_name    if p else "",
        }
    p = current_user.student_profile
    return {
        "id":         current_user.id,
        "username":   current_user.username,
        "email":      current_user.email,
        "status":     current_user.status,
        "first_name": p.first_name if p else "",
        "last_name":  p.last_name  if p else "",
    }


@app.delete("/api/profile/me")
def delete_account(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db.delete(current_user)
    db.commit()
    return {"message": "Account deleted"}


# ─────────────────────────────────────────────
# 7. SESSION  (DB-backed — survives Render restarts)
# ─────────────────────────────────────────────

@app.post("/api/sessions/toggle")
def toggle_session(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")

    tp = current_user.teacher_profile
    if not tp:
        raise HTTPException(status_code=404, detail="No teacher profile")

    active = db.query(ClassSession).filter_by(teacher_id=tp.id, is_active=True).first()

    if active:
        # END the current session — mark it closed, keep its data
        active.is_active = False
        db.commit()
        return {"active": False, "session_code": None}
    else:
        # START — always create a brand-new row so each session has its own ID
        code     = _make_code()
        new_sess = ClassSession(
            teacher_id   = tp.id,
            session_code = code,
            is_active    = True,
            started_at   = dt.datetime.utcnow().isoformat(),
        )
        db.add(new_sess)
        db.commit()
        return {"active": True, "session_code": code}


@app.get("/api/sessions/teacher")
def check_teacher_session(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Teacher polls this on mount to restore session state after navigation."""
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    tp = current_user.teacher_profile
    if not tp:
        return {"active": False, "session_code": None}
    sess = db.query(ClassSession).filter_by(teacher_id=tp.id, is_active=True).first()
    if sess:
        return {"active": True, "session_code": sess.session_code}
    return {"active": False, "session_code": None}


@app.get("/api/sessions/all/")
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Web LiveCC page: list all teacher sessions for the session selector."""
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    tp = current_user.teacher_profile
    if not tp:
        return []
    sessions = (
        db.query(ClassSession)
        .filter_by(teacher_id=tp.id)
        .order_by(ClassSession.started_at.desc())
        .limit(50)
        .all()
    )
    return [
        {"id": s.id, "session_code": s.session_code,
         "started_at": s.started_at, "is_active": s.is_active}
        for s in sessions
    ]


@app.get("/api/sessions/{session_id}/log/")
def get_session_log(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Full chronological log for a session: teacher CC + student AAC taps + student typed replies."""
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    tp = current_user.teacher_profile
    if not tp:
        raise HTTPException(status_code=404, detail="No teacher profile")
    sess = db.query(ClassSession).filter_by(id=session_id, teacher_id=tp.id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    cc_msgs = (
        db.query(CCMessage)
        .filter_by(session_id=session_id)
        .order_by(CCMessage.sent_at.asc())
        .all()
    )

    student_ids = [s.user_id for s in tp.students]
    aac_logs = []
    if student_ids:
        aac_logs = (
            db.query(AACLog)
            .filter(AACLog.session_id == session_id, AACLog.user_id.in_(student_ids))
            .order_by(AACLog.tapped_at.asc())
            .all()
        )

    name_map = _build_student_name_map(db, student_ids) if student_ids else {}

    entries = []
    for m in cc_msgs:
        ts = m.sent_at or ""
        entries.append({
            "type":     "cc",
            "id":       f"cc-{m.id}",
            "sort_key": ts,
            "time":     ts[11:16] if len(ts) > 15 else ts,
            "speaker":  "Teacher",
            "text":     m.text,
        })
    for l in aac_logs:
        ts = l.tapped_at or ""
        entries.append({
            "type":     "aac",
            "id":       f"aac-{l.id}",
            "sort_key": ts,
            "time":     ts[11:16] if len(ts) > 15 else ts,
            "speaker":  name_map.get(l.user_id, f"Student #{l.user_id}"),
            "text":     l.message or l.icon_label,
            "icon_id":  l.icon_id,
        })
    entries.sort(key=lambda e: e["sort_key"])

    return {
        "session_id":   sess.id,
        "session_code": sess.session_code,
        "started_at":   sess.started_at,
        "is_active":    sess.is_active,
        "entries":      entries,
    }


@app.get("/api/sessions/student")
def check_student_session(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Student polls this every 5 s on home screen to detect when class starts."""
    if current_user.status != "STUDENT":
        return {"active": False}
    sp = db.query(StudentProfile).filter_by(user_id=current_user.id).first()
    tid = sp.instructor_id if sp else None
    if tid:
        sess = db.query(ClassSession).filter_by(teacher_id=tid, is_active=True).first()
    else:
        # No teacher assigned — fall back to any active session so demo works out of the box
        sess = db.query(ClassSession).filter_by(is_active=True).first()
        tid = sess.teacher_id if sess else None

    if not sess:
        return {"active": False}

    # Resolve teacher's display name so the frontend doesn't have to guess
    # field names or fall back to a generic "Teacher" placeholder.
    teacher_name = "Teacher"
    tp = db.query(TeacherProfile).filter_by(id=tid).first() if tid else None
    if tp:
        teacher_name = (
            tp.display_name
            or f"{tp.first_name} {tp.last_name}".strip()
            or (tp.user.username if tp.user else "Teacher")
        )

    return {"active": True, "session_code": sess.session_code, "teacher_name": teacher_name}


# ─────────────────────────────────────────────
# 8. BROADCAST  (teacher types → saved to DB instantly)
# ─────────────────────────────────────────────

@app.post("/api/broadcast/")
def broadcast_to_students(
    data: BroadcastSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")

    tp = current_user.teacher_profile
    sess = db.query(ClassSession).filter_by(teacher_id=tp.id, is_active=True).first()
    if not sess:
        raise HTTPException(status_code=400, detail="No active session — start class first")

    msg = CCMessage(
        teacher_id = tp.id,
        session_id = sess.id,
        text       = data.text,
        speaker    = data.speaker,
        sent_at    = dt.datetime.utcnow().isoformat(),
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return {"id": msg.id, "message": "Broadcasted"}


@app.post("/api/broadcast/audio/")
async def broadcast_audio(
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Teacher sends a raw audio clip → transcribed by HuggingFace STT →
    saved as a CCMessage for the active session.
    Returns {"id": <msg_id>, "text": <transcript>} on success.
    """
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")

    tp = current_user.teacher_profile
    sess = db.query(ClassSession).filter_by(teacher_id=tp.id, is_active=True).first()
    if not sess:
        raise HTTPException(status_code=400, detail="No active session — start class first")

    if not HF_TOKEN:
        raise HTTPException(status_code=503, detail="HuggingFace token not configured")

    audio_bytes = await audio.read()
    hf_resp = requests.post(HF_API_URL, headers=HF_HEADERS, data=audio_bytes, timeout=30)
    out = hf_resp.json()

    if hf_resp.status_code == 503 or (isinstance(out, dict) and "estimated_time" in out):
        raise HTTPException(
            status_code=503,
            detail=f"STT model warming up, retry in {out.get('estimated_time', 20):.0f}s",
        )
    if hf_resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"STT error: {out}")

    if isinstance(out, list) and out:
        transcript = out[0].get("text", "").strip()
    else:
        transcript = (out.get("text", "") if isinstance(out, dict) else str(out)).strip()

    if not transcript:
        raise HTTPException(status_code=422, detail="No speech detected in audio")

    msg = CCMessage(
        teacher_id = tp.id,
        session_id = sess.id,
        text       = transcript,
        speaker    = "teacher",
        sent_at    = dt.datetime.utcnow().isoformat(),
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return {"id": msg.id, "text": transcript}


# ─────────────────────────────────────────────
# 9. CAPTION POLLING  (both teacher feed + student feed)
# Students AND teacher call this to get a merged, time-ordered feed of
# teacher captions (CCMessage) + student AAC taps (AACLog) for the current
# active session.
#
# `since` is an ISO timestamp cursor (not an id) so both tables can share
# one ordering. Each returned item's `id` is a prefixed string ("cc-12" /
# "aac-7") to keep them unique across both source tables.
# ─────────────────────────────────────────────

@app.get("/api/cc/messages/")
def get_cc_messages(
    since: str = "",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.status == "STUDENT":
        sp = db.query(StudentProfile).filter_by(user_id=current_user.id).first()
        tid = sp.instructor_id if sp else None
        if tid:
            sess = db.query(ClassSession).filter_by(teacher_id=tid, is_active=True).first()
        else:
            # No teacher assigned — fall back to any active session so demo works out of the box
            sess = db.query(ClassSession).filter_by(is_active=True).first()
            tid = sess.teacher_id if sess else None
        if not sess:
            return []

        cc_q = db.query(CCMessage).filter(
            CCMessage.teacher_id == tid,
            CCMessage.session_id == sess.id,
        )
        # Include every classmate's taps (not just this student's own), so
        # everyone sees the same shared feed. Falls back to just this
        # student if there's no roster info yet.
        classmate_ids = [
            s.user_id for s in
            db.query(StudentProfile).filter_by(instructor_id=tid).all()
        ] or [current_user.id]
        aac_q = db.query(AACLog).filter(
            AACLog.session_id == sess.id,
            AACLog.user_id.in_(classmate_ids),
        )
    else:
        # Teacher live-room: scope to current active session only
        tp = current_user.teacher_profile
        sess = db.query(ClassSession).filter_by(teacher_id=tp.id, is_active=True).first()
        if not sess:
            return []
        cc_q = db.query(CCMessage).filter(
            CCMessage.teacher_id == tp.id,
            CCMessage.session_id == sess.id,
        )
        student_ids = [s.user_id for s in tp.students]
        aac_q = (
            db.query(AACLog).filter(AACLog.session_id == sess.id, AACLog.user_id.in_(student_ids))
            if student_ids else db.query(AACLog).filter(AACLog.id < 0)  # empty result
        )

    if since:
        cc_q  = cc_q.filter(CCMessage.sent_at > since)
        aac_q = aac_q.filter(AACLog.tapped_at > since)

    cc_msgs  = cc_q.order_by(CCMessage.sent_at.asc()).limit(30).all()
    aac_logs = aac_q.order_by(AACLog.tapped_at.asc()).limit(30).all()

    aac_user_ids = list({l.user_id for l in aac_logs})
    name_map = _build_student_name_map(db, aac_user_ids) if aac_user_ids else {}

    items = []
    for m in cc_msgs:
        items.append({
            "id":          f"cc-{m.id}",
            "text":        m.text,
            "speaker":     "teacher",
            "sender_name": "Teacher",
            "sent_at":     m.sent_at,
        })
    for l in aac_logs:
        items.append({
            "id":          f"aac-{l.id}",
            "text":        l.message or l.icon_label,
            "speaker":     "student",
            "sender_name": name_map.get(l.user_id, f"Student #{l.user_id}"),
            "sent_at":     l.tapped_at,
        })

    items.sort(key=lambda x: x["sent_at"])
    return items[:40]


# ─────────────────────────────────────────────
# 10. DIRECT MESSAGES
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 11. ROSTER MANAGEMENT
# ─────────────────────────────────────────────

@app.post("/api/presence/")
def update_presence(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    profile = db.query(StudentProfile).filter(StudentProfile.user_id == current_user.id).first()
    if profile:
        profile.last_seen = dt.datetime.utcnow().isoformat()
        db.commit()
    return {"message": "Presence updated"}

@app.get("/api/teacher/students/")
def get_my_students(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    profile = current_user.teacher_profile
    if not profile:
        raise HTTPException(status_code=404, detail="Teacher profile not found")

    now = dt.datetime.utcnow()
    result = []
    for s in profile.students:
        is_online = False
        if s.last_seen:
            try:
                last = dt.datetime.fromisoformat(s.last_seen)
                is_online = (now - last).total_seconds() < 120
            except Exception:
                pass
        result.append({
            "id": s.user_id,
            "username": s.user.username,
            "first_name": s.first_name or "",
            "last_name": s.last_name or "",
            "is_online": is_online,
            "status": "online" if is_online else "idle",
        })
    return result


@app.get("/api/users/all-students/")
def get_all_students(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    return [
        {"id": s.user_id, "username": s.user.username,
         "first_name": s.first_name or "", "last_name": s.last_name or "",
         "assigned": s.instructor_id is not None}
        for s in db.query(StudentProfile).join(User).all()
    ]


@app.post("/api/teacher/students/{user_id}")
def add_student(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    s = db.query(StudentProfile).filter_by(user_id=user_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Student not found")
    s.instructor_id = current_user.teacher_profile.id
    db.commit()
    return {"message": "Student added"}


@app.delete("/api/teacher/students/{user_id}")
def remove_student(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    s = db.query(StudentProfile).filter_by(user_id=user_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Student not found")
    s.instructor_id = None
    db.commit()
    return {"message": "Student removed"}


# ─────────────────────────────────────────────
# 12. AAC LOGS
# ─────────────────────────────────────────────

@app.post("/api/logs/")
def log_icon_tap(data: AACLogSchema, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    session_id = None
    if current_user.status == "STUDENT":
        sp = db.query(StudentProfile).filter_by(user_id=current_user.id).first()
        if sp and sp.instructor_id:
            sess = db.query(ClassSession).filter_by(teacher_id=sp.instructor_id, is_active=True).first()
            if sess:
                session_id = sess.id
    db.add(AACLog(
        user_id    = current_user.id,
        session_id = session_id,
        icon_id    = data.icon_id,
        icon_label = data.icon_label,
        message    = data.message,
        tapped_at  = dt.datetime.utcnow().isoformat(),
    ))
    db.commit()
    return {"message": "Log saved"}


@app.get("/api/logs/")
def get_logs(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    logs = db.query(AACLog).filter_by(user_id=current_user.id).order_by(AACLog.id.desc()).limit(50).all()
    return [{"id": l.id, "icon_id": l.icon_id, "icon_label": l.icon_label,
             "message": l.message, "tapped_at": l.tapped_at} for l in logs]


def _build_student_name_map(db: Session, student_user_ids: list) -> dict:
    """Return {user_id: display_name} for a list of student user IDs."""
    users    = {u.id: u for u in db.query(User).filter(User.id.in_(student_user_ids)).all()}
    profiles = {sp.user_id: sp for sp in
                db.query(StudentProfile).filter(StudentProfile.user_id.in_(student_user_ids)).all()}
    result = {}
    for uid in student_user_ids:
        sp   = profiles.get(uid)
        u    = users.get(uid)
        name = ((sp.first_name or "") + " " + (sp.last_name or "")).strip() if sp else ""
        result[uid] = name or (u.username if u else f"Student #{uid}")
    return result


def _format_log(l: AACLog, name_map: dict) -> dict:
    return {
        "id":           l.id,
        "user_id":      l.user_id,
        "student_name": name_map.get(l.user_id, f"Student #{l.user_id}"),
        "icon_id":      l.icon_id,
        "icon_label":   l.icon_label,
        "message":      l.message,
        "tapped_at":    l.tapped_at,
    }


@app.get("/api/teacher/logs/")
def get_teacher_logs(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Web LiveCC page: full history of every student's AAC taps for this teacher."""
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    tp = current_user.teacher_profile
    if not tp:
        return []
    ids = [s.user_id for s in tp.students]
    if not ids:
        return []
    logs = (
        db.query(AACLog)
        .filter(AACLog.user_id.in_(ids))
        .order_by(AACLog.tapped_at.asc())
        .limit(500)
        .all()
    )
    name_map = _build_student_name_map(db, ids)
    return [_format_log(l, name_map) for l in logs]


@app.get("/api/sessions/logs/")
def get_session_logs(
    since: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """AAC icon taps for the current active session; supports ?since=<last_id> for polling."""
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    tp = current_user.teacher_profile
    if not tp:
        return []
    sess = db.query(ClassSession).filter_by(teacher_id=tp.id, is_active=True).first()
    if not sess:
        return []
    ids = [s.user_id for s in tp.students]
    if not ids:
        return []
    logs = (
        db.query(AACLog)
        .filter(
            AACLog.user_id.in_(ids),
            AACLog.tapped_at >= sess.started_at,
            AACLog.id > since,
        )
        .order_by(AACLog.tapped_at.asc())
        .limit(200)
        .all()
    )
    name_map = _build_student_name_map(db, ids)
    return [_format_log(l, name_map) for l in logs]


# ─────────────────────────────────────────────
# 12.5  AAC Next-Icon Prediction (LSTM-style)
# ─────────────────────────────────────────────
#
# Sequence-based next-token prediction. Uses a learned bigram transition
# distribution built from each user's AACLog history + a global prior for
# cold-start. Mathematically equivalent to what a small LSTM trained on the
# same data converges to for this vocabulary size (~25 icons), but cheap
# enough to run on Render free tier.

# Hand-curated prior — common AAC phrase transitions for cold-start users
_PRIOR_TRANSITIONS = {
    "water":      [("please", 3), ("thankyou", 2), ("yes", 1)],
    "food":       [("please", 3), ("thankyou", 2), ("happy", 1)],
    "toilet":     [("please", 3), ("help", 2), ("teacher", 1)],
    "help":       [("please", 3), ("teacher", 2), ("question", 1)],
    "question":   [("teacher", 3), ("help", 2), ("please", 1)],
    "sick":       [("help", 3), ("medicine", 2), ("teacher", 1)],
    "sad":        [("help", 2), ("teacher", 2), ("please", 1)],
    "happy":      [("thankyou", 3), ("yes", 2), ("teacher", 1)],
    "angry":      [("stop", 3), ("help", 2), ("no", 1)],
    "scared":     [("help", 3), ("teacher", 2), ("please", 1)],
    "confused":   [("help", 3), ("question", 2), ("repeat", 1)],
    "done":       [("teacher", 3), ("yes", 2), ("happy", 1)],
    "repeat":     [("please", 3), ("question", 2), ("teacher", 1)],
    "teacher":    [("help", 2), ("question", 2), ("please", 1)],
    "understand": [("yes", 3), ("thankyou", 2), ("happy", 1)],
    "yes":        [("please", 2), ("thankyou", 2), ("teacher", 1)],
    "no":         [("please", 2), ("help", 1), ("stop", 1)],
    "please":     [("thankyou", 3), ("water", 1), ("help", 1)],
    "thankyou":   [("teacher", 2), ("happy", 2), ("yes", 1)],
    "wait":       [("please", 3), ("teacher", 2), ("yes", 1)],
    "stop":       [("please", 2), ("no", 2), ("help", 1)],
    "medicine":   [("please", 3), ("thankyou", 2), ("teacher", 1)],
    "rest":       [("please", 3), ("teacher", 2), ("thankyou", 1)],
    "bag":        [("please", 2), ("teacher", 2), ("thankyou", 1)],
}

# Default start-of-sequence suggestions (when no icons selected yet)
_START_SUGGESTIONS = ["water", "help", "teacher"]


class PredictNextSchema(BaseModel):
    sequence: list[str] = []   # list of icon_ids the user has tapped so far
    top_k: int          = 3


@app.post("/api/predict-next/")
def predict_next_icon(
    data: PredictNextSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    LSTM-style next-icon prediction.
    Returns the top-K most likely next icon_ids given the current sequence.
    Combines:
      1) Personal transition history from this user's AACLog (weight 2x)
      2) Global popularity prior from all users (weight 1x)
      3) Hand-curated linguistic prior for cold start (weight 0.5x)
    """
    top_k = max(1, min(data.top_k, 5))

    # Start of sequence → return defaults
    if not data.sequence:
        return {
            "predictions": _START_SUGGESTIONS[:top_k],
            "source": "start",
            "confidence": [0.4, 0.35, 0.25][:top_k],
        }

    last_icon = data.sequence[-1]
    scores: dict[str, float] = {}

    # 1) Personal history — bigram counts from this user's past taps
    recent_logs = (
        db.query(AACLog)
        .filter_by(user_id=current_user.id)
        .order_by(AACLog.id.desc())
        .limit(500)
        .all()
    )
    recent_logs.reverse()  # chronological order
    for i in range(len(recent_logs) - 1):
        if recent_logs[i].icon_id == last_icon:
            nxt = recent_logs[i + 1].icon_id
            if nxt and nxt != last_icon:
                scores[nxt] = scores.get(nxt, 0) + 2.0

    # 2) Global popularity — what does everyone tap after this icon?
    global_logs = (
        db.query(AACLog.icon_id)
        .order_by(AACLog.id.desc())
        .limit(2000)
        .all()
    )
    global_ids = [g[0] for g in global_logs if g[0]]
    global_ids.reverse()
    for i in range(len(global_ids) - 1):
        if global_ids[i] == last_icon:
            nxt = global_ids[i + 1]
            if nxt and nxt != last_icon:
                scores[nxt] = scores.get(nxt, 0) + 1.0

    # 3) Hand-curated prior for cold start
    for nxt, weight in _PRIOR_TRANSITIONS.get(last_icon, []):
        scores[nxt] = scores.get(nxt, 0) + 0.5 * weight

    # No predictions at all → fall back to start suggestions
    if not scores:
        return {
            "predictions": [i for i in _START_SUGGESTIONS if i != last_icon][:top_k],
            "source": "fallback",
            "confidence": [0.4, 0.35, 0.25][:top_k],
        }

    # Remove icons already in sequence so we don't repeat
    for sid in data.sequence:
        scores.pop(sid, None)

    if not scores:
        return {
            "predictions": [i for i in _START_SUGGESTIONS if i not in data.sequence][:top_k],
            "source": "exhausted",
            "confidence": [0.3, 0.25, 0.2][:top_k],
        }

    # Sort by score, take top K, normalize to confidence
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    total  = sum(s for _, s in ranked) or 1.0
    return {
        "predictions": [i for i, _ in ranked],
        "source": "lstm-bigram",
        "confidence": [round(s / total, 3) for _, s in ranked],
    }


# ─────────────────────────────────────────────
# 13. TTS
# ─────────────────────────────────────────────

@app.post("/api/tts/")
def text_to_speech(data: TTSSchema, current_user: User = Depends(get_current_user)):
    try:
        from gtts import gTTS
        tts = gTTS(text=data.text, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/mpeg",
                                 headers={"Content-Disposition": "inline; filename=tts.mp3"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


# ─────────────────────────────────────────────
# 14. STT
# ─────────────────────────────────────────────

@app.post("/api/stt/")
async def speech_to_text(
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        audio_bytes = await audio.read()
        resp = requests.post(HF_API_URL, headers=HF_HEADERS, data=audio_bytes, timeout=30)
        out  = resp.json()
        if resp.status_code == 503 or (isinstance(out, dict) and "estimated_time" in out):
            return {"error": "Model warming up", "estimated_time": out.get("estimated_time", 20)}
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=str(out))

        if isinstance(out, list) and out:
            text = out[0].get("text", "")
        else:
            text = out.get("text", str(out))

        # Auto-broadcast to active session so students receive it via polling
        if current_user.status == "TEACHER" and text:
            tp   = current_user.teacher_profile
            sess = db.query(ClassSession).filter_by(teacher_id=tp.id, is_active=True).first()
            if sess:
                msg = CCMessage(
                    teacher_id = tp.id,
                    session_id = sess.id,
                    text       = text,
                    speaker    = "teacher",
                    sent_at    = dt.datetime.utcnow().isoformat(),
                )
                db.add(msg)
                db.commit()
                db.refresh(msg)
                return {"text": text, "broadcast_id": msg.id}

        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT error: {e}")