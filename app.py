import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import sqlite3
import threading
import os
import re
import base64
import json
import requests
import time
from pathlib import Path
from datetime import datetime, timedelta
#from dotenv import load_dotenv, dotenv_values
import asyncio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# BASE CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
#load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MODEL_NAME     = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.5-pro")
DB_PATH        = BASE_DIR / "homecare_memory.db"
BG_PATH        = BASE_DIR / "uploads" / "ui" / "assets" / "homecare_bg.png"
LOGO_DIR       = BASE_DIR / "uploads" / "ui" / "logo"
LOGO_DIR.mkdir(parents=True, exist_ok=True)
LOGO_PATH      = LOGO_DIR / "company_logo.png"

# ← مسار الملفات الجديدة (مرن للـ Windows والـ Server)
_win_output = Path(r"E:\moshb analysis\custmer\Mohab_AI_OS\الملفات الجديده")
try:
    _win_output.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR = _win_output
except Exception:
    OUTPUT_DIR = BASE_DIR / "الملفات الجديده"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = BASE_DIR / "تقارير"
REPORTS_DIR.mkdir(exist_ok=True)

# ============================================================
# TELEGRAM CONFIG
# ============================================================
def parse_telegram_from_env():
    env_vars = st.secrets
    bots = {}
    for key, value in env_vars.items():
        if key.startswith("TELEGRAM_BOT_") and value and len(value) > 10:
            bots[key.replace("TELEGRAM_BOT_", "").lower()] = value
    return bots

TELEGRAM_BOTS      = parse_telegram_from_env()
ADMIN_TELEGRAM_TOKEN = os.getenv("ADMIN_TELEGRAM_TOKEN",
    list(TELEGRAM_BOTS.values())[0] if TELEGRAM_BOTS else "")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "")

def send_telegram_message(token: str, chat_id: str, message: str) -> bool:
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message[:4000], "parse_mode": "HTML"},
            timeout=10
        )
        return r.status_code == 200
    except Exception:
        return False

def notify_admin_startup(username: str):
    if not ADMIN_TELEGRAM_TOKEN or not ADMIN_CHAT_ID:
        return
    msg = (f"🏠 <b>Home Care AI OS</b>\n━━━━━━━━━━━━━━━\n"
           f"👤 <b>المستخدم:</b> {username}\n"
           f"🕐 <b>الوقت:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
           f"📡 <b>الحالة:</b> شغّال ✅\n━━━━━━━━━━━━━━━\n💡 النظام نشط الآن")
    threading.Thread(target=send_telegram_message,
                     args=(ADMIN_TELEGRAM_TOKEN, ADMIN_CHAT_ID, msg), daemon=True).start()

# ============================================================
# AI SETUP — يدعم google-generativeai و google-genai
# ============================================================
def normalize_model_name(name: str) -> str:
    return name[len("models/"):] if name.startswith("models/") else name

MODEL_NAME = normalize_model_name(MODEL_NAME)

# ← محاولة google.genai أولاً ثم google.generativeai كـ fallback
_genai_new = False
client     = None
model_obj  = None
model_available = False
model_error     = None

try:
    from google import genai as _gnai
    from google.genai import types as _gnai_types
    client = _gnai.Client(api_key=GOOGLE_API_KEY)
    client.models.get(model=MODEL_NAME)
    model_available = True
    _genai_new = True
except Exception as e1:
    try:
        import google.generativeai as _gai
        _gai.configure(api_key=GOOGLE_API_KEY)
        model_obj = _gai.GenerativeModel(MODEL_NAME)
        model_available = True
        _genai_new = False
    except Exception as e2:
        model_error = f"google.genai: {e1} | generativeai: {e2}"

# Auto-select model if current failed
if not model_available:
    try:
        import google.generativeai as _gai
        _gai.configure(api_key=GOOGLE_API_KEY)
        res = _gai.list_models()
        names = [normalize_model_name(getattr(m,"name",str(m))) for m in res]
        for pattern in ["2.5-pro","2.5-flash","1.5-pro","pro","gemini"]:
            chosen = next((n for n in names if pattern in n.lower()), None)
            if chosen:
                model_obj = _gai.GenerativeModel(chosen)
                MODEL_NAME = chosen
                model_available = True
                _genai_new = False
                break
    except Exception:
        pass

def ask_ai_raw(prompt_text: str, system_hint: str = "") -> str:
    if not model_available:
        return "⚠️ الـ AI غير متاح — تأكد من GOOGLE_API_KEY"
    full = f"{system_hint}\n\n{prompt_text}" if system_hint else prompt_text
    full = re.sub(r"\s+", " ", full.strip())
    try:
        if _genai_new and client:
            response = client.models.generate_content(model=MODEL_NAME, contents=full)
            text = getattr(response, "text", None)
            if not text:
                try: text = response.candidates[0].content.parts[0].text
                except: text = str(response)
        else:
            response = model_obj.generate_content(full)
            text = getattr(response, "text", None) or str(response)
        return (text or "").strip()
    except Exception as e:
        return f"⚠️ خطأ: {e}"

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="HOME CARE AI OS",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Home Care AI OS — Dual Agent Intelligence"}
)

# ============================================================
# DATABASE
# ============================================================
_db_lock = threading.Lock()

def get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL, source TEXT DEFAULT 'web',
                role TEXT NOT NULL, content TEXT NOT NULL,
                intent TEXT DEFAULT 'عام', created_at TEXT NOT NULL
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT, master_sheet TEXT, file_path TEXT,
                created_at TEXT, file_size INTEGER DEFAULT 0,
                sheet_names TEXT DEFAULT '', description TEXT DEFAULT '',
                sheet_relationships TEXT DEFAULT ''
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL, role TEXT DEFAULT 'user',
                email TEXT DEFAULT '', created_at TEXT, last_login TEXT,
                is_active INTEGER DEFAULT 1, department TEXT DEFAULT '',
                personality_notes TEXT DEFAULT '', kpi_context TEXT DEFAULT ''
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT, report_name TEXT, format TEXT,
                file_path TEXT, created_at TEXT
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS user_intelligence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                communication_style TEXT DEFAULT 'عام',
                preferred_language TEXT DEFAULT 'مصري',
                frequent_topics TEXT DEFAULT '',
                personality_summary TEXT DEFAULT '',
                last_updated TEXT
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS self_learning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight TEXT NOT NULL, category TEXT DEFAULT 'عام',
                confidence REAL DEFAULT 0.5, applied_count INTEGER DEFAULT 0,
                created_at TEXT, approved INTEGER DEFAULT 0, approved_by TEXT DEFAULT ''
            )""")
            conn.commit()
        finally:
            conn.close()

def migrate_database():
    with _db_lock:
        conn = get_conn()
        try:
            for table, col, defn in [
                ("memory","intent","TEXT DEFAULT 'عام'"),
                ("files","sheet_relationships","TEXT DEFAULT ''"),
                ("files","description","TEXT DEFAULT ''"),
                ("users","personality_notes","TEXT DEFAULT ''"),
                ("users","kpi_context","TEXT DEFAULT ''"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")
                    conn.commit()
                except Exception:
                    pass
        finally:
            conn.close()

init_database()
migrate_database()

# ── Admin User ──
def init_admin_user():
    with _db_lock:
        conn = get_conn()
        try:
            if not conn.execute("SELECT id FROM users WHERE username=?", ("مهاب",)).fetchone():
                conn.execute(
                    "INSERT INTO users (username,role,email,created_at,last_login,department) VALUES(?,?,?,?,?,?)",
                    ("مهاب","admin","mohab@homecare.ai",
                     datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), "إدارة")
                )
                conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

init_admin_user()

# ── DB Functions ──
def save_memory(user_id, source, role, content, intent="عام"):
    content = content.strip()
    if not content: return
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("INSERT INTO memory (user_id,source,role,content,intent,created_at) VALUES(?,?,?,?,?,?)",
                         (user_id, source, role, content, intent, datetime.utcnow().isoformat()))
            conn.commit()
        finally:
            conn.close()

def load_memory(user_id, limit=20):
    with _db_lock:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT role,content FROM memory WHERE user_id=? ORDER BY rowid DESC LIMIT ?",
                (user_id, limit)).fetchall()
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
        finally:
            conn.close()

def load_memory_timeline(user_id, limit=50):
    """للـ AI Memory Timeline"""
    with _db_lock:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT role,content,intent,created_at FROM memory WHERE user_id=? ORDER BY rowid DESC LIMIT ?",
                (user_id, limit)).fetchall()
            return [{"role": r[0], "content": r[1], "intent": r[2], "created_at": r[3]}
                    for r in reversed(rows)]
        finally:
            conn.close()

def save_file_metadata(file_name, master_sheet, file_path, sheet_names, description="", relationships=""):
    with _db_lock:
        conn = get_conn()
        try:
            fsize = 0
            try:
                if Path(file_path).exists(): fsize = Path(file_path).stat().st_size
            except: pass
            conn.execute(
                "INSERT INTO files (file_name,master_sheet,file_path,created_at,file_size,sheet_names,description,sheet_relationships) VALUES(?,?,?,?,?,?,?,?)",
                (file_name, master_sheet, str(file_path), datetime.utcnow().isoformat(),
                 fsize, ",".join(sheet_names), description, relationships))
            conn.commit()
        finally:
            conn.close()

def get_files_history(limit=10):
    with _db_lock:
        conn = get_conn()
        try:
            return conn.execute("SELECT id,file_name,master_sheet,created_at FROM files ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        finally:
            conn.close()

def create_user(username, role="user", email=""):
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("INSERT INTO users (username,role,email,created_at) VALUES(?,?,?,?)",
                         (username, role, email, datetime.utcnow().isoformat()))
            conn.commit()
            return True
        except: return False
        finally: conn.close()

def get_user(username):
    with _db_lock:
        conn = get_conn()
        try:
            row = conn.execute("SELECT id,role,email,created_at FROM users WHERE username=?", (username,)).fetchone()
            return {"id": row[0], "username": username, "role": row[1], "email": row[2], "created_at": row[3]} if row else None
        finally: conn.close()

def update_user_login(username):
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("UPDATE users SET last_login=? WHERE username=?", (datetime.utcnow().isoformat(), username))
            conn.commit()
        finally: conn.close()

def list_all_users():
    with _db_lock:
        conn = get_conn()
        try:
            return conn.execute("SELECT username,role,email,created_at FROM users WHERE is_active=1 ORDER BY created_at DESC").fetchall()
        finally: conn.close()

def save_export(user_id, report_name, format_type, file_path):
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("INSERT INTO exports (user_id,report_name,format,file_path,created_at) VALUES(?,?,?,?,?)",
                         (user_id, report_name, format_type, str(file_path), datetime.utcnow().isoformat()))
            conn.commit()
        finally: conn.close()

def update_user_intelligence(user_id, style, topics, summary):
    with _db_lock:
        conn = get_conn()
        try:
            if conn.execute("SELECT id FROM user_intelligence WHERE user_id=?", (user_id,)).fetchone():
                conn.execute(
                    "UPDATE user_intelligence SET communication_style=?,frequent_topics=?,personality_summary=?,last_updated=? WHERE user_id=?",
                    (style, topics, summary, datetime.utcnow().isoformat(), user_id))
            else:
                conn.execute(
                    "INSERT INTO user_intelligence (user_id,communication_style,frequent_topics,personality_summary,last_updated) VALUES(?,?,?,?,?)",
                    (user_id, style, topics, summary, datetime.utcnow().isoformat()))
            conn.commit()
        finally: conn.close()

def get_user_intelligence(user_id):
    with _db_lock:
        conn = get_conn()
        try:
            row = conn.execute("SELECT communication_style,frequent_topics,personality_summary FROM user_intelligence WHERE user_id=?", (user_id,)).fetchone()
            return {"style": row[0], "topics": row[1], "summary": row[2]} if row else None
        finally: conn.close()

def save_self_learning_insight(insight, category, confidence=0.7):
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("INSERT INTO self_learning (insight,category,confidence,created_at) VALUES(?,?,?,?)",
                         (insight, category, confidence, datetime.utcnow().isoformat()))
            conn.commit()
        finally: conn.close()

def get_pending_insights(limit=5):
    with _db_lock:
        conn = get_conn()
        try:
            return conn.execute("SELECT id,insight,category,confidence FROM self_learning WHERE approved=0 ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        finally: conn.close()

def approve_insight(insight_id, approved_by):
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("UPDATE self_learning SET approved=1,approved_by=? WHERE id=?", (approved_by, insight_id))
            conn.commit()
        finally: conn.close()

def get_user_reports_dir(username):
    path = OUTPUT_DIR / username / datetime.now().strftime("%Y-%m-%d")
    path.mkdir(parents=True, exist_ok=True)
    return path

# ============================================================
# AI PROFILES
# ============================================================
MOHAB_PROFILE = """
اسمك: مهاب يونس
أنت محلل بيانات وخبير تشغيل أعمال: تحليل مالي، إدارة مخزون، Excel/PowerBI، أتمتة.
أسلوبك: عملي، سريع، دقيق — تفهم المصري والبلدي والفكاهي والفصحى وكل أنواع الكلام.
مهامك: تحليل Excel/XLSM، تلخيص، توصيات تنفيذية، نظام ذاكرة يتذكر سياق العمل.
تعليمات: اجب كـ"مهاب" — لا تشكر، كن مباشراً، اعطِ أرقام حقيقية من البيانات.
"""

ANALYST_PROFILE = """
اسمك: وكيل التحليل العميق
أنت متخصص في: قراءة وربط الشيتات، تحليل KPI، اكتشاف الأنماط، التوقعات الاستراتيجية.
أسلوبك: دقيق، تحليلي، يعطي أرقام وحقائق — لا آراء فقط.
"""

# ============================================================
# LANGUAGE INTELLIGENCE
# ============================================================
LANGUAGE_PATTERNS = {
    "تحليل":   ["حلل","اتحلل","عايز تحليل","ايه اللي بيحصل","فيه ايه","ايه الحكاية","analyze"],
    "مخزون":   ["مخزن","مخزون","بضاعة","stock","inventory","عندنا ايه"],
    "مبيعات":  ["مبيعات","بيع","اتباع","sales","revenue","فلوس","حجينا"],
    "توقع":    ["متوقع","هيحصل ايه","forecast","predict","المستقبل","بكره"],
    "مشكلة":   ["مشكلة","فيه حاجة غلط","خطأ","مش شغال","ليه ده"],
    "kpi":     ["kpi","أداء","performance","مؤشرات","نتايج","معدلات"],
    "تقرير":   ["تقرير","report","ملخص","summary","جهزلي","عايز ورقة"],
    "مقارنة":  ["قارن","الفرق","compare","مقارنة","احسن من","اسوأ من"],
    "تليجرام": ["ابعت","send","تليجرام","telegram","ارسل","وصّل"],
    "كود":     ["كود","code","رقم","حالة","status","item"],
    "فكاهي":   ["يلا","يابن الحلال","مش معقول","عالي","تحفة","جامد","ازيك"],
    "عميل":    ["عميل","customer","client","زبون","مشتري"],
    "مالي":    ["فلوس","مالية","cash","نقدية","دفع","سداد"],
}

def detect_intent(text):
    text_lower = text.lower()
    intents = [k for k, pats in LANGUAGE_PATTERNS.items() if any(p in text_lower for p in pats)]
    return intents or ["عام"]

# ============================================================
# SMART SCORES — Inventory Health, Customer Risk, etc.
# ============================================================
def calculate_inventory_health_score(df: pd.DataFrame) -> dict:
    """حساب Inventory Health Score من 0 إلى 100"""
    score = 100
    details = []
    qty_keywords = ["qty","quantity","كمية","مخزون","stock","balance","رصيد","عدد"]
    price_keywords = ["price","سعر","cost","تكلفة","value","قيمة"]
    
    qty_col = next((c for c in df.columns if any(k in str(c).lower() for k in qty_keywords)), None)
    price_col = next((c for c in df.columns if any(k in str(c).lower() for k in price_keywords)), None)
    
    if qty_col:
        qty = df[qty_col].fillna(0)
        zero_pct = (qty == 0).sum() / max(len(qty), 1) * 100
        low_pct  = ((qty > 0) & (qty <= qty.quantile(0.2))).sum() / max(len(qty), 1) * 100
        
        if zero_pct > 20:
            score -= 25
            details.append(f"🔴 {zero_pct:.1f}% أصناف نفدت (0 كمية)")
        elif zero_pct > 10:
            score -= 15
            details.append(f"🟡 {zero_pct:.1f}% أصناف قاربت النفاد")
        else:
            details.append(f"✅ نسبة نفاد المخزون منخفضة ({zero_pct:.1f}%)")
        
        if low_pct > 30:
            score -= 15
            details.append(f"⚠️ {low_pct:.1f}% أصناف بمخزون منخفض")
        
        cv = qty.std() / max(qty.mean(), 1)
        if cv > 2:
            score -= 10
            details.append(f"⚠️ تشتت كبير في الكميات (CV={cv:.1f})")
    
    missing_pct = df.isna().sum().sum() / max(df.size, 1) * 100
    if missing_pct > 10:
        score -= 10
        details.append(f"⚠️ {missing_pct:.1f}% بيانات ناقصة")
    
    score = max(0, min(100, score))
    
    if score >= 80: color, label = "#10b981", "ممتاز"
    elif score >= 60: color, label = "#f59e0b", "متوسط"
    elif score >= 40: color, label = "#f97316", "يحتاج انتباه"
    else: color, label = "#ef4444", "خطر"
    
    return {"score": score, "color": color, "label": label, "details": details}

def calculate_customer_risk_score(df: pd.DataFrame) -> dict:
    """Customer Risk Score"""
    score = 50
    details = []
    
    sales_keywords = ["مبيعات","sales","revenue","amount","total","قيمة","مبلغ"]
    sales_col = next((c for c in df.select_dtypes(include='number').columns
                      if any(k in str(c).lower() for k in sales_keywords)), None)
    
    if sales_col:
        vals = df[sales_col].dropna()
        if len(vals) > 0:
            cv = vals.std() / max(vals.mean(), 1)
            if cv > 1.5:
                score += 20
                details.append(f"⚠️ تباين كبير في المبيعات (مؤشر مخاطرة)")
            else:
                score -= 10
                details.append("✅ مبيعات مستقرة نسبياً")
            
            q25 = vals.quantile(0.25)
            low_customers = (vals < q25).sum()
            pct = low_customers / max(len(vals), 1) * 100
            if pct > 40:
                score += 15
                details.append(f"⚠️ {pct:.0f}% من العملاء بمبيعات منخفضة جداً")
    
    score = max(0, min(100, score))
    
    if score <= 30: color, label = "#10b981", "منخفض"
    elif score <= 60: color, label = "#f59e0b", "متوسط"
    elif score <= 80: color, label = "#f97316", "مرتفع"
    else: color, label = "#ef4444", "خطر عالٍ"
    
    return {"score": score, "color": color, "label": label, "details": details}

def calculate_cashflow_health(df: pd.DataFrame) -> dict:
    """Cash Flow Health Score"""
    score = 60
    details = []
    
    cash_keywords = ["cash","نقد","حساب","balance","رصيد","مدفوع","paid"]
    cash_col = next((c for c in df.select_dtypes(include='number').columns
                     if any(k in str(c).lower() for k in cash_keywords)), None)
    
    if cash_col:
        vals = df[cash_col].dropna()
        if len(vals) > 0:
            neg_pct = (vals < 0).sum() / max(len(vals), 1) * 100
            if neg_pct > 0:
                score -= neg_pct * 0.5
                details.append(f"⚠️ {neg_pct:.1f}% قيم سالبة (عجز)")
            else:
                score += 15
                details.append("✅ لا توجد قيم سالبة")
    
    score = max(0, min(100, score))
    
    if score >= 70: color, label = "#10b981", "صحي"
    elif score >= 50: color, label = "#f59e0b", "مقبول"
    else: color, label = "#ef4444", "يحتاج مراجعة"
    
    return {"score": score, "color": color, "label": label, "details": details}

def calculate_stock_prediction_score(df: pd.DataFrame) -> dict:
    """Stock Prediction Confidence"""
    score = 50
    details = []
    
    qty_keywords = ["qty","quantity","كمية","stock","مخزون"]
    qty_col = next((c for c in df.select_dtypes(include='number').columns
                    if any(k in str(c).lower() for k in qty_keywords)), None)
    
    if qty_col and len(df) > 5:
        vals = df[qty_col].dropna()
        trend = np.polyfit(range(len(vals)), vals, 1)[0] if len(vals) > 2 else 0
        if trend < 0:
            score -= abs(trend / max(vals.mean(), 1)) * 100
            details.append(f"📉 اتجاه تنازلي — توقع نقص في المخزون")
        else:
            score += 10
            details.append("📈 اتجاه إيجابي")
        
        score += min(len(vals) / 10, 30)
        details.append(f"📊 ثقة التنبؤ مبنية على {len(vals)} سجل")
    
    score = max(0, min(100, score))
    
    if score >= 70: color, label = "#10b981", "ثقة عالية"
    elif score >= 50: color, label = "#f59e0b", "ثقة متوسطة"
    else: color, label = "#ef4444", "ثقة منخفضة"
    
    return {"score": score, "color": color, "label": label, "details": details}

def calculate_ai_confidence(prompt_len: int, data_rows: int, response_len: int) -> dict:
    """AI Confidence % بناءً على جودة المدخلات"""
    score = 50
    if prompt_len > 50: score += 10
    if data_rows > 100: score += 20
    elif data_rows > 10: score += 10
    if response_len > 200: score += 15
    score = min(95, score)
    
    if score >= 80: color, label = "#10b981", "عالي"
    elif score >= 60: color, label = "#f59e0b", "متوسط"
    else: color, label = "#ef4444", "منخفض"
    
    return {"score": score, "color": color, "label": label}

# ============================================================
# SHEET INTELLIGENCE
# ============================================================
def detect_sheet_type(sheet_name, df):
    name = sheet_name.lower()
    if any(x in name for x in ["مبيع","sales","بيع","فاتور"]): return "مبيعات"
    if any(x in name for x in ["مخزن","مخزون","stock","inventory"]): return "مخزون"
    if any(x in name for x in ["location","لوكيشن","فرع","branch"]): return "فروع"
    if any(x in name for x in ["عميل","customer","client"]): return "عملاء"
    if any(x in name for x in ["موظف","employee","staff","hr"]): return "موارد بشرية"
    if any(x in name for x in ["مالي","finance","حساب"]): return "مالي"
    if any(x in name for x in ["كود","code","master","رئيسي"]): return "بيانات رئيسية"
    cols = " ".join([str(c).lower() for c in df.columns])
    if any(x in cols for x in ["qty","quantity","كمية"]): return "مخزون/مبيعات"
    return "بيانات عامة"

def find_sheet_relationships(sheets_dict):
    rels = {}
    keys = list(sheets_dict.keys())
    for a in keys:
        r = []
        for b in keys:
            if a == b: continue
            common = set(str(c) for c in sheets_dict[a].columns) & set(str(c) for c in sheets_dict[b].columns)
            if common:
                r.append({"sheet": b, "common_columns": list(common), "strength": len(common)})
        if r:
            rels[a] = sorted(r, key=lambda x: x["strength"], reverse=True)
    return rels

def build_deep_sheet_context(sheets_dict, master_sheet, sub_sheets):
    rels = find_sheet_relationships(sheets_dict)
    mdf  = sheets_dict[master_sheet]
    ctx  = f"\n🧠 تحليل هيكل الملف ({len(sheets_dict)} شيت):\n"
    ctx += f"📍 الشيت الأساسي: {master_sheet} ({len(mdf)} صف | {detect_sheet_type(master_sheet, mdf)})\n"
    ctx += f"   الأعمدة: {list(mdf.columns[:15])}\n"
    
    for sn, df in sheets_dict.items():
        if sn == master_sheet: continue
        ctx += f"\n📋 {sn} ({detect_sheet_type(sn,df)}): {len(df)} صف | {list(df.columns[:8])}"
        for col in df.select_dtypes(include='number').columns[:3]:
            try: ctx += f"\n   {col}: Σ={df[col].sum():.2f} | μ={df[col].mean():.2f}"
            except: pass
    
    ctx += "\n\n🔗 العلاقات:\n"
    for sheet, srels in rels.items():
        if srels:
            top = srels[0]
            ctx += f"  {sheet} ↔ {top['sheet']} عبر {top['common_columns'][:3]}\n"
    return ctx

# ============================================================
# AI FUNCTIONS
# ============================================================
def build_ai_prompt(user_input, data_context, memory_text, user_id,
                    master_context="", user_intel=None, intents=None):
    intel_section = ""
    if user_intel:
        intel_section = f"\n🧠 ذاكرة المستخدم: أسلوب={user_intel.get('style','عام')} | مواضيع={user_intel.get('topics','')}\n"
    intent_section = f"🎯 نية: {', '.join(intents or ['عام'])}\n" if intents else ""

    return f"""
SYSTEM: {MOHAB_PROFILE}
{ANALYST_PROFILE}

أنت Dual Agent: مهاب (واجهة + لهجة) + وكيل التحليل (عمق + أرقام).
قواعد: لا تشكر، كن مباشراً، أعطِ أرقام حقيقية، اقترح خطوات تنفيذية.

USER_ID: {user_id}
{intent_section}{intel_section}
{master_context}

سجل المحادثة:
{memory_text}

البيانات:
{data_context}

طلب المستخدم:
{user_input}

اردّ بذكاء وعمق. اذكر أرقام حقيقية. في النهاية: الخطوة التالية المقترحة.
"""

def ask_analyst_agent(data_context, question, master_context=""):
    prompt = f"""
{ANALYST_PROFILE}
البيانات:
{master_context}
{data_context}

السؤال: {question}

قدم:
1. الأرقام والإحصاءات الحقيقية
2. اكتشاف الأنماط والمشاكل
3. مقارنات وتوقعات
4. توصيات قابلة للتنفيذ
5. KPIs مقترحة
"""
    return ask_ai_raw(prompt, ANALYST_PROFILE)

def dual_agent_response(user_input, data_context, memory_text,
                        user_id, master_context="", user_intel=None):
    intents = detect_intent(user_input)
    analyst_insight = ""
    
    if data_context and any(i in intents for i in ["تحليل","مخزون","مبيعات","kpi","مقارنة","توقع","كود","عميل","مالي"]):
        analyst_insight = ask_analyst_agent(data_context, user_input, master_context)
    
    full_prompt = build_ai_prompt(user_input, data_context, memory_text,
                                   user_id, master_context, user_intel, intents)
    if analyst_insight:
        full_prompt += f"\n\n[وكيل التحليل]:\n{analyst_insight[:600]}"
    
    response = ask_ai_raw(full_prompt, MOHAB_PROFILE)
    
    # Self-learning
    if len(user_input) > 20 and data_context:
        try:
            ins = ask_ai_raw(f"من هذا التفاعل، استخرج درساً مفيداً واحداً (جملة واحدة عربي فقط):\nسؤال: {user_input[:100]}\nرد: {response[:150]}")
            if ins and 5 < len(ins) < 200:
                save_self_learning_insight(ins, ", ".join(intents))
        except Exception:
            pass
    
    return response, analyst_insight

# ============================================================
# CHARTS
# ============================================================
CHART_CFG = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font_color="#e2e8f0", grid="#ffffff0d")

def style_fig(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#60a5fa"), x=0.02),
        paper_bgcolor=CHART_CFG["paper_bgcolor"],
        plot_bgcolor=CHART_CFG["plot_bgcolor"],
        font=dict(color=CHART_CFG["font_color"], family="Cairo, sans-serif", size=10),
        margin=dict(l=15, r=15, t=40, b=15),
        legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="rgba(37,99,235,0.3)",
                    borderwidth=1, font=dict(size=9)),
    )
    fig.update_xaxes(gridcolor=CHART_CFG["grid"], linecolor="#ffffff10", tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=CHART_CFG["grid"], linecolor="#ffffff10", tickfont=dict(size=9))
    return fig

def auto_generate_charts(df, sheet_name, question=""):
    charts = []
    if df is None or df.empty: return charts
    
    num = df.select_dtypes(include='number').columns.tolist()
    cat = df.select_dtypes(include=['object','category']).columns.tolist()
    date_cols = [c for c in df.columns if any(x in str(c).lower() for x in ['date','تاريخ','يوم','شهر','time'])]
    
    colors = ["#2563eb","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4"]
    
    # Bar
    if cat and num:
        try:
            grouped = df.groupby(cat[0])[num[0]].sum().reset_index().sort_values(num[0], ascending=False).head(12)
            fig = px.bar(grouped, x=cat[0], y=num[0], color=num[0], color_continuous_scale="Blues", text=num[0])
            fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            charts.append(style_fig(fig, f"🏆 أعلى 12 — {num[0]} حسب {cat[0]}"))
        except: pass
    
    # Donut
    if cat and num:
        try:
            pie_data = df.groupby(cat[0])[num[0]].sum().reset_index().head(8)
            fig = px.pie(pie_data, names=cat[0], values=num[0],
                         color_discrete_sequence=px.colors.qualitative.Set3, hole=0.45)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            charts.append(style_fig(fig, f"🍩 التوزيع — {num[0]}"))
        except: pass
    
    # Line trend
    if date_cols and num:
        try:
            dtmp = df[[date_cols[0], num[0]]].dropna().copy()
            dtmp[date_cols[0]] = pd.to_datetime(dtmp[date_cols[0]], errors='coerce')
            dtmp = dtmp.dropna().sort_values(date_cols[0])
            if len(dtmp) > 2:
                fig = px.line(dtmp, x=date_cols[0], y=num[0], color_discrete_sequence=["#10b981"], markers=True)
                fig.update_traces(line=dict(width=2.5))
                charts.append(style_fig(fig, f"📈 الاتجاه — {num[0]}"))
        except: pass
    
    # Histogram
    if num:
        try:
            fig = px.histogram(df, x=num[0], nbins=20, color_discrete_sequence=["#2563eb"])
            charts.append(style_fig(fig, f"📊 توزيع — {num[0]}"))
        except: pass
    
    # Scatter
    if len(num) >= 2:
        try:
            fig = px.scatter(df, x=num[0], y=num[1],
                             color=cat[0] if cat else None,
                             color_discrete_sequence=colors, opacity=0.7)
            charts.append(style_fig(fig, f"🔵 {num[0]} vs {num[1]}"))
        except: pass
    
    # Heatmap
    if len(num) >= 3:
        try:
            corr = df[num[:8]].corr()
            fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                                        colorscale='RdBu', zmid=0,
                                        text=corr.round(2).values, texttemplate="%{text}", showscale=True))
            charts.append(style_fig(fig, "🔥 مصفوفة الارتباط"))
        except: pass
    
    return charts

def generate_kpi_dashboard(sheets_dict, master_sheet):
    if not sheets_dict or master_sheet not in sheets_dict: return None
    df = sheets_dict[master_sheet]
    num = df.select_dtypes(include='number').columns.tolist()
    if not num: return None
    
    n = min(len(num), 4)
    fig = make_subplots(rows=1, cols=n, specs=[[{"type":"indicator"}]*n])
    colors = ["#2563eb","#10b981","#f59e0b","#ef4444"]
    
    for i, col in enumerate(num[:n]):
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=df[col].sum(),
            title={"text": col, "font": {"size": 11, "color": "#94a3b8"}},
            number={"font": {"color": colors[i], "size": 30}, "valueformat": ",.0f"},
            delta={"reference": df[col].mean() * max(len(df)*0.85,1), "relative": True, "font": {"size": 10}},
        ), row=1, col=i+1)
    
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0", family="Cairo"),
                      height=150, margin=dict(l=5, r=5, t=25, b=5))
    return fig

# ============================================================
# EXPORT FUNCTIONS
# ============================================================
def export_to_pdf(title, content, user_id, filename=None):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        
        fn = filename or f"{title.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        fp = get_user_reports_dir(user_id) / fn
        doc = SimpleDocTemplate(str(fp), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(f"📊 {title}", ParagraphStyle('T', parent=styles['Heading1'],
                fontSize=20, textColor=colors.HexColor('#1d4ed8'), spaceAfter=20, alignment=1)),
            Spacer(1, 0.2*inch),
            Paragraph(f"<b>التاريخ:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']),
            Paragraph(f"<b>المستخدم:</b> {user_id}", styles['Normal']),
            Spacer(1, 0.3*inch),
        ]
        for line in content.split('\n'):
            line = line.strip()
            if line:
                try:
                    safe = line.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
                    story.append(Paragraph(safe, styles['Normal']))
                    story.append(Spacer(1, 0.04*inch))
                except: pass
        doc.build(story)
        save_export(user_id, title, "PDF", str(fp))
        return fp
    except ImportError:
        st.warning("⚠️ pip install reportlab")
        return None
    except Exception as e:
        st.error(f"خطأ PDF: {e}")
        return None

def export_to_excel(title, data_dict, user_id, filename=None):
    try:
        fn = filename or f"{title.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        fp = get_user_reports_dir(user_id) / fn
        with pd.ExcelWriter(str(fp), engine='openpyxl') as writer:
            for sname, data in data_dict.items():
                (data if isinstance(data, pd.DataFrame) else pd.DataFrame([data])).to_excel(
                    writer, sheet_name=sname[:31], index=False)
        save_export(user_id, title, "Excel", str(fp))
        return fp
    except Exception as e:
        st.error(f"خطأ Excel: {e}")
        return None

# ============================================================
# IMAGE HELPERS
# ============================================================
def get_base64_image(path):
    try:
        if Path(path).exists():
            return base64.b64encode(Path(path).read_bytes()).decode()
    except: pass
    return None

def read_all_sheets(uploaded_file):
    try:
        ef = pd.ExcelFile(uploaded_file)
        return {sn: pd.read_excel(uploaded_file, sheet_name=sn) for sn in ef.sheet_names}
    except: return None

# ============================================================
# SESSION STATE
# ============================================================
_defaults = {
    "current_user": "مهاب",
    "messages": [],
    "df": None,
    "sheets_dict": None,
    "master_sheet": None,
    "sub_sheets": [],
    "last_prompt": "",
    "last_response": None,
    "last_analyst": None,
    "telegram_started": False,
    "show_charts": True,
    "music_on": False,
    "music_track": 0,
    "startup_notified": False,
    "pending_send_telegram": None,
    "ask_save_report": False,
    "last_charts": [],
    "show_memory_timeline": False,
    "page_loaded": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.startup_notified:
    notify_admin_startup(st.session_state.current_user)
    st.session_state.startup_notified = True

# ============================================================
# CSS — Awwwards Level + Animations + Skeleton + Transitions
# ============================================================
bg_b64   = get_base64_image(BG_PATH)
bg_css   = f"url('data:image/png;base64,{bg_b64}')" if bg_b64 else "none"
logo_b64 = get_base64_image(LOGO_PATH)
logo_html = (
    f"<img src='data:image/png;base64,{logo_b64}' style='width:48px;height:48px;object-fit:contain;border-radius:10px;'/>"
    if logo_b64 else
    "<div style='width:48px;height:48px;background:linear-gradient(135deg,#dc2626,#f97316);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:26px;'>🏠</div>"
)

css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');

*, *::before, *::after {{
    font-family: 'Cairo', sans-serif !important;
    box-sizing: border-box;
}}

/* ─── PAGE TRANSITION ─── */
.stApp {{
    animation: pageEnter 0.6s cubic-bezier(0.16,1,0.3,1) both;
    background-image: linear-gradient(rgba(3,7,26,0.93), rgba(3,7,26,0.93)), {bg_css};
    background-size: cover;
    background-attachment: fixed;
    color: #e2e8f0;
    min-height: 100vh;
}}
@keyframes pageEnter {{
    from {{ opacity:0; transform:translateY(12px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}

/* ─── SIDEBAR ─── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,rgba(5,10,28,0.99) 0%,rgba(10,15,38,0.99) 100%) !important;
    border-right: 1px solid rgba(37,99,235,0.2) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.6);
}}

/* ─── MAIN ─── */
.block-container {{
    padding: 1.4rem 2rem 2rem 2rem !important;
    max-width: 1500px;
}}

/* ─── HEADER ─── */
.hc-header {{
    display:flex; align-items:center; gap:16px;
    padding:18px 24px;
    background:rgba(15,23,42,0.85);
    border:1px solid rgba(37,99,235,0.28);
    border-radius:18px; margin-bottom:10px;
    backdrop-filter:blur(16px);
    box-shadow:0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
    animation: fadeSlideDown 0.5s ease both;
}}
@keyframes fadeSlideDown {{
    from {{ opacity:0; transform:translateY(-10px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
.hc-title {{
    font-size:1.5rem !important;
    font-weight:900 !important;
    background: linear-gradient(135deg,#dc2626,#f97316,#ea580c);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    margin:0 !important; line-height:1.2;
    letter-spacing:-0.5px;
}}
.hc-sub {{
    font-size:0.75rem !important;
    color:#475569 !important; margin:2px 0 0 0 !important;
}}

/* ─── SCORE CARDS (Odoo / Stripe style) ─── */
.score-grid {{
    display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:14px 0;
}}
.score-card {{
    background:rgba(15,23,42,0.88);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:14px; padding:14px 12px;
    position:relative; overflow:hidden;
    transition:all 0.3s cubic-bezier(0.16,1,0.3,1);
    cursor:default;
}}
.score-card::before {{
    content:'';
    position:absolute; top:0; left:0; right:0; height:2px;
    background:var(--score-color,#2563eb);
    border-radius:14px 14px 0 0;
}}
.score-card:hover {{
    transform:translateY(-3px);
    box-shadow:0 12px 40px rgba(0,0,0,0.4);
    border-color:var(--score-color,rgba(37,99,235,0.4));
}}
.score-value {{
    font-size:2rem; font-weight:900; color:var(--score-color,#2563eb);
    line-height:1; margin-bottom:3px;
    animation: countUp 1.5s ease both;
}}
@keyframes countUp {{
    from {{ opacity:0; transform:scale(0.7); }}
    to   {{ opacity:1; transform:scale(1); }}
}}
.score-label {{ font-size:0.7rem; color:#64748b; margin-bottom:4px; }}
.score-badge {{
    display:inline-block;
    background:rgba(var(--score-rgb,37,99,235),0.12);
    border:1px solid var(--score-color,#2563eb);
    border-radius:20px; padding:1px 8px;
    font-size:0.65rem; color:var(--score-color,#2563eb); font-weight:700;
}}
.score-bar {{
    height:3px; background:rgba(255,255,255,0.06); border-radius:2px;
    margin-top:8px; overflow:hidden;
}}
.score-bar-fill {{
    height:100%; background:var(--score-color,#2563eb);
    border-radius:2px;
    animation: barFill 1.5s cubic-bezier(0.16,1,0.3,1) both;
    transform-origin:left;
}}
@keyframes barFill {{
    from {{ transform:scaleX(0); }}
    to   {{ transform:scaleX(1); }}
}}

/* ─── SKELETON SCREENS ─── */
.skeleton {{
    background: linear-gradient(90deg,rgba(255,255,255,0.04) 25%,rgba(255,255,255,0.08) 50%,rgba(255,255,255,0.04) 75%);
    background-size:200% 100%;
    animation:shimmer 1.8s infinite;
    border-radius:8px;
}}
@keyframes shimmer {{
    0%   {{ background-position:200% 0; }}
    100% {{ background-position:-200% 0; }}
}}

/* ─── CHAT BUBBLES ─── */
.chat-section-title {{
    font-size:0.95rem; font-weight:700; color:#e2e8f0;
    margin:18px 0 10px 0; display:flex; align-items:center; gap:8px;
}}
.chat-section-title::after {{
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,rgba(37,99,235,0.4),transparent);
}}
.chat-user-wrapper {{
    display:flex; justify-content:flex-end; margin-bottom:14px;
    animation:fadeSlideRight 0.3s ease both;
}}
.chat-user-bubble {{
    background:linear-gradient(135deg,#2563eb 0%,#1d4ed8 100%);
    color:#fff; padding:13px 18px;
    border-radius:18px 4px 18px 18px;
    max-width:72%; box-shadow:0 4px 24px rgba(37,99,235,0.3);
    font-size:0.9rem; line-height:1.7;
    border:1px solid rgba(96,165,250,0.2);
    word-break:break-word;
}}
.chat-user-label {{ font-size:0.7rem; color:rgba(255,255,255,0.55); text-align:right; margin-bottom:4px; }}

.chat-assistant-wrapper {{
    display:flex; justify-content:flex-start; margin-bottom:14px;
    animation:fadeSlideLeft 0.35s ease both;
}}
.chat-assistant-bubble {{
    background:rgba(15,23,42,0.93);
    color:#e2e8f0; padding:14px 18px;
    border-radius:4px 18px 18px 18px;
    max-width:78%; box-shadow:0 4px 24px rgba(0,0,0,0.35);
    font-size:0.9rem; line-height:1.8;
    border:1px solid rgba(100,116,139,0.22);
    word-break:break-word;
}}
.chat-assistant-label {{
    font-size:0.7rem; color:#60a5fa; margin-bottom:6px;
    display:flex; align-items:center; gap:5px;
}}
.agent-tag {{
    background:rgba(37,99,235,0.15); border:1px solid rgba(37,99,235,0.3);
    border-radius:4px; padding:1px 6px; font-size:0.62rem; color:#93c5fd;
}}
.analyst-tag {{
    background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3);
    border-radius:4px; padding:1px 6px; font-size:0.62rem; color:#6ee7b7;
}}

/* ─── AI THINKING ANIMATION ─── */
.ai-thinking {{
    display:flex; align-items:center; gap:8px; padding:12px 18px;
    background:rgba(15,23,42,0.85);
    border:1px solid rgba(37,99,235,0.2);
    border-radius:4px 18px 18px 18px; max-width:200px;
    margin-bottom:14px;
}}
.thinking-dots {{ display:flex; gap:5px; }}
.thinking-dot {{
    width:8px; height:8px; border-radius:50%;
    background:#2563eb; animation:thinkPulse 1.4s infinite;
}}
.thinking-dot:nth-child(2){{ animation-delay:0.2s; background:#f59e0b; }}
.thinking-dot:nth-child(3){{ animation-delay:0.4s; background:#10b981; }}
@keyframes thinkPulse {{
    0%,80%,100%{{ transform:scale(0.7); opacity:0.5; }}
    40%{{ transform:scale(1.2); opacity:1; }}
}}
.thinking-text {{ font-size:0.78rem; color:#64748b; }}

/* ─── MEMORY TIMELINE ─── */
.timeline-container {{ position:relative; padding:10px 0; }}
.timeline-line {{
    position:absolute; left:22px; top:0; bottom:0;
    width:2px; background:linear-gradient(180deg,rgba(37,99,235,0.5),transparent);
}}
.timeline-item {{
    display:flex; gap:16px; margin-bottom:16px;
    animation:fadeSlideLeft 0.3s ease both;
}}
.timeline-dot {{
    width:16px; height:16px; border-radius:50%;
    border:2px solid #2563eb; background:#0f172a;
    flex-shrink:0; margin-top:3px; z-index:1;
    position:relative;
    box-shadow:0 0 10px rgba(37,99,235,0.4);
}}
.timeline-dot.user {{ border-color:#f59e0b; box-shadow:0 0 10px rgba(245,158,11,0.4); }}
.timeline-content {{
    background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.05);
    border-radius:10px; padding:10px 14px; flex:1;
}}
.timeline-meta {{ font-size:0.68rem; color:#475569; margin-bottom:4px; }}
.timeline-text {{ font-size:0.82rem; color:#cbd5e1; line-height:1.6; }}
.timeline-intent {{
    display:inline-block; background:rgba(37,99,235,0.12);
    border:1px solid rgba(37,99,235,0.25); border-radius:12px;
    padding:1px 8px; font-size:0.62rem; color:#60a5fa; margin-top:4px;
}}

/* ─── WELCOME ─── */
.welcome-card {{
    background:rgba(15,23,42,0.85);
    border:1px dashed rgba(37,99,235,0.3);
    border-radius:22px; padding:32px 28px;
    text-align:center; margin:20px auto; max-width:620px;
    backdrop-filter:blur(10px);
    animation:fadeSlideDown 0.5s ease 0.2s both;
}}
.welcome-card h2 {{ font-size:1.1rem; color:#e2e8f0; margin-bottom:10px; }}
.welcome-card p  {{ font-size:0.83rem; color:#64748b; line-height:2; }}

/* ─── ANALYST INSIGHT ─── */
.analyst-insight {{
    background:rgba(16,185,129,0.06);
    border:1px solid rgba(16,185,129,0.2);
    border-radius:12px; padding:12px 16px; margin-top:8px;
    font-size:0.83rem; color:#a7f3d0;
}}

/* ─── SCROLL REVEAL ─── */
.reveal {{
    animation:scrollReveal 0.6s cubic-bezier(0.16,1,0.3,1) both;
}}
@keyframes scrollReveal {{
    from {{ opacity:0; transform:translateY(20px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}

/* ─── INSIGHT BADGE ─── */
.insight-badge {{
    background:rgba(16,185,129,0.09);
    border:1px solid rgba(16,185,129,0.3);
    border-radius:8px; padding:6px 10px;
    font-size:0.78rem; color:#34d399; margin:3px 0;
}}

/* ─── SAVE PROMPT ─── */
.save-prompt {{
    background:rgba(15,23,42,0.92);
    border:1px solid rgba(245,158,11,0.3);
    border-radius:16px; padding:16px 20px; margin:10px 0;
}}
.save-title {{ font-size:0.9rem; font-weight:700; color:#fbbf24; margin-bottom:10px; }}

/* ─── SHEET BADGES ─── */
.sheet-badge {{ display:inline-flex; align-items:center; gap:5px; padding:3px 10px; border-radius:20px; font-size:0.7rem; font-weight:700; margin:2px; }}
.sheet-master {{ background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.35); color:#fbbf24; }}
.sheet-sub    {{ background:rgba(37,99,235,0.1); border:1px solid rgba(37,99,235,0.25); color:#60a5fa; }}

/* ─── MUSIC BAR ─── */
.music-bar {{
    background:rgba(15,23,42,0.9); border:1px solid rgba(37,99,235,0.2);
    border-radius:10px; padding:7px 12px; margin:5px 0;
    font-size:0.75rem; color:#94a3b8;
}}

/* ─── BUTTONS ─── */
.stButton>button {{
    background:linear-gradient(135deg,#2563eb,#1d4ed8) !important;
    color:#fff !important; border:none !important;
    border-radius:10px !important; font-weight:700 !important;
    font-size:0.82rem !important; padding:8px 16px !important;
    transition:all 0.25s ease !important;
}}
.stButton>button:hover {{
    background:linear-gradient(135deg,#1d4ed8,#1e3a8a) !important;
    box-shadow:0 4px 20px rgba(37,99,235,0.45) !important;
    transform:translateY(-1px) !important;
}}

/* ─── INPUTS ─── */
.stTextInput>div>div>input,
div[data-testid="stChatInput"] textarea {{
    background:rgba(15,23,42,0.95) !important;
    color:#e2e8f0 !important;
    border:1px solid rgba(37,99,235,0.3) !important;
    border-radius:12px !important; font-size:0.9rem !important;
}}
div[data-testid="stChatInput"] textarea:focus {{
    border-color:rgba(37,99,235,0.65) !important;
    box-shadow:0 0 0 3px rgba(37,99,235,0.12) !important;
}}
.stSelectbox>div, .stMultiSelect>div {{
    background:rgba(15,23,42,0.9) !important;
    border-color:rgba(37,99,235,0.25) !important;
    border-radius:10px !important;
}}
hr {{ border-color:rgba(37,99,235,0.12) !important; margin:10px 0 !important; }}
.streamlit-expanderHeader {{
    background:rgba(15,23,42,0.8) !important;
    border-radius:10px !important;
    border:1px solid rgba(37,99,235,0.18) !important;
    color:#94a3b8 !important; font-size:0.83rem !important;
}}
[data-testid="metric-container"] {{
    background:rgba(15,23,42,0.85) !important;
    border:1px solid rgba(37,99,235,0.18) !important;
    border-radius:12px !important; padding:12px !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size:1.45rem !important; color:#2563eb !important; font-weight:900 !important;
}}

/* ─── ANIMATIONS ─── */
@keyframes fadeSlideRight {{ from{{ opacity:0;transform:translateX(14px); }} to{{ opacity:1;transform:translateX(0); }} }}
@keyframes fadeSlideLeft  {{ from{{ opacity:0;transform:translateX(-14px); }} to{{ opacity:1;transform:translateX(0); }} }}
@keyframes pulse {{ 0%,100%{{ opacity:1; }} 50%{{ opacity:0.45; }} }}

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar {{ width:5px; }}
::-webkit-scrollbar-track {{ background:rgba(15,23,42,0.4); }}
::-webkit-scrollbar-thumb {{ background:rgba(37,99,235,0.35); border-radius:3px; }}
::-webkit-scrollbar-thumb:hover {{ background:rgba(37,99,235,0.6); }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ============================================================
# PREMIUM CSS ADDITIONS — Aurora + Neon + Bento + Particles
# ============================================================
st.markdown("""
<style>

/* ─── AURORA ANIMATED BACKGROUND ─── */
@keyframes auroraShift {
    0%   { background-position:0% 50%; }
    50%  { background-position:100% 50%; }
    100% { background-position:0% 50%; }
}

/* ─── NEON GLOW TEXT ─── */
.neon-text {
    color:#00d4ff;
    text-shadow: 0 0 10px #00d4ff, 0 0 20px #00d4ff, 0 0 30px #667eea, 0 0 40px #667eea;
}

/* ─── BENTO GRID ─── */
.bento-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));
    gap:16px; margin:20px 0;
}
.bento-item {
    background:linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,41,59,0.8));
    border:1px solid rgba(0,212,255,0.15);
    border-radius:18px; padding:24px;
    position:relative; overflow:hidden;
    animation:bentoReveal 0.6s cubic-bezier(0.16,1,0.3,1) both;
    box-shadow:0 8px 32px rgba(0,0,0,0.2);
    transition:all 0.3s ease;
}
@keyframes bentoReveal {
    from { opacity:0; transform:scale(0.9) translateY(12px); }
    to   { opacity:1; transform:scale(1) translateY(0); }
}
.bento-item:hover {
    transform:translateY(-4px);
    box-shadow:0 12px 48px rgba(0,212,255,0.15);
    border-color:rgba(0,212,255,0.3);
}
.bento-item::before {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg, transparent, rgba(0,212,255,0.4), transparent);
}

/* ─── PREMIUM HEADER SHINE ─── */
.hc-header { position:relative; overflow:hidden; }
.hc-header::before {
    content:''; position:absolute; top:0; left:-100%;
    width:200%; height:2px;
    background:linear-gradient(90deg, transparent, #00d4ff, transparent);
    animation:headerShine 7s ease-in-out infinite;
}
@keyframes headerShine {
    0%   { left:-100%; }
    50%  { left:100%; }
    100% { left:100%; }
}
.hc-title {
    text-shadow:0 0 30px rgba(220,38,38,0.4), 0 0 60px rgba(249,115,22,0.2) !important;
}

/* ─── GLASS PANEL ─── */
.glass-panel {
    background:rgba(15,23,42,0.88);
    backdrop-filter:blur(20px);
    border:1px solid rgba(255,255,255,0.08);
    box-shadow:0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.08);
    border-radius:18px; padding:20px;
}

/* ─── PREMIUM SCORE CARD ENHANCEMENTS ─── */
.score-card {
    background:linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.85)) !important;
    box-shadow:0 0 20px rgba(0,212,255,0.08), inset 0 1px 0 rgba(255,255,255,0.06) !important;
}
.score-card:hover {
    box-shadow:0 16px 48px rgba(0,0,0,0.4), 0 0 30px rgba(0,212,255,0.15) !important;
}
.score-value {
    animation:scoreCountUp 1.2s cubic-bezier(0.34,1.56,0.64,1) both !important;
}
@keyframes scoreCountUp {
    from { opacity:0; transform:scale(0.3) translateY(12px); }
    to   { opacity:1; transform:scale(1) translateY(0); }
}

/* ─── ANIMATED METRIC CARDS ─── */
.metric-card {
    background:linear-gradient(135deg, rgba(37,99,235,0.08), rgba(16,185,129,0.06));
    border:1px solid rgba(0,212,255,0.12);
    border-radius:16px; padding:18px;
    position:relative; overflow:hidden;
    transition:all 0.3s ease;
    text-align:center;
}
.metric-card:hover {
    border-color:rgba(0,212,255,0.3);
    transform:translateY(-3px);
    box-shadow:0 12px 32px rgba(0,212,255,0.1);
}
.metric-card::before {
    content:''; position:absolute; top:-50%; right:-50%; width:200%; height:200%;
    background:radial-gradient(circle, rgba(0,212,255,0.06), transparent);
    animation:metricPulse 5s ease-in-out infinite;
}
@keyframes metricPulse {
    0%,100% { transform:translate(0,0) scale(1); opacity:0.3; }
    50%     { transform:translate(20px,20px) scale(1.2); opacity:0.6; }
}

/* ─── PREMIUM BUBBLE ENHANCEMENTS ─── */
.chat-user-bubble {
    box-shadow:0 8px 32px rgba(37,99,235,0.25), 0 0 30px rgba(37,99,235,0.12) !important;
    backdrop-filter:blur(20px) !important;
}
.chat-assistant-bubble {
    border:1px solid rgba(0,212,255,0.18) !important;
    box-shadow:0 8px 32px rgba(0,0,0,0.35), 0 0 20px rgba(0,212,255,0.06) !important;
    backdrop-filter:blur(20px) !important;
}

/* ─── AI THINKING STREAM ─── */
.ai-thinking-stream {
    display:flex; align-items:center; gap:12px; padding:16px 20px;
    background:linear-gradient(135deg, rgba(0,212,255,0.05), rgba(102,126,234,0.05));
    border:1px solid rgba(0,212,255,0.2);
    border-radius:4px 18px 18px 18px;
    max-width:300px; margin-bottom:14px;
    animation:thinkingFloat 0.5s ease both;
}
@keyframes thinkingFloat {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
.thinking-dot {
    background:radial-gradient(circle at 30% 30%, #00d4ff, #667eea) !important;
    box-shadow:0 0 10px rgba(0,212,255,0.5) !important;
}
.thinking-dot:nth-child(2) {
    background:radial-gradient(circle at 30% 30%, #667eea, #f093fb) !important;
    box-shadow:0 0 10px rgba(102,126,234,0.5) !important;
}
.thinking-dot:nth-child(3) {
    background:radial-gradient(circle at 30% 30%, #f093fb, #10b981) !important;
    box-shadow:0 0 10px rgba(240,147,251,0.5) !important;
}

/* ─── PREMIUM BUTTON SHINE ─── */
.stButton>button {
    box-shadow:0 0 20px rgba(37,99,235,0.25) !important;
    position:relative; overflow:hidden;
}
.stButton>button::before {
    content:''; position:absolute; top:0; left:-100%;
    width:100%; height:100%;
    background:linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    animation:btnShine 7s infinite;
}
@keyframes btnShine {
    0%   { left:-100%; }
    50%  { left:100%; }
    100% { left:100%; }
}
.stButton>button:hover {
    transform:translateY(-3px) !important;
    box-shadow:0 8px 32px rgba(37,99,235,0.5) !important;
}

/* ─── PREMIUM INPUT ENHANCEMENT ─── */
.stTextInput>div>div>input,
div[data-testid="stChatInput"] textarea {
    background:linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.85)) !important;
    border:1px solid rgba(0,212,255,0.2) !important;
    box-shadow:inset 0 2px 8px rgba(0,0,0,0.3) !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color:rgba(0,212,255,0.6) !important;
    box-shadow:inset 0 2px 8px rgba(0,0,0,0.3), 0 0 0 3px rgba(0,212,255,0.12) !important;
}

/* ─── TIMELINE PREMIUM GLOW ─── */
.timeline-dot {
    box-shadow:0 0 20px rgba(37,99,235,0.5), inset 0 0 10px rgba(255,255,255,0.08) !important;
}
.timeline-dot.user {
    box-shadow:0 0 20px rgba(245,158,11,0.5), inset 0 0 10px rgba(255,255,255,0.08) !important;
}

/* ─── INSIGHT BADGE GLOW ─── */
.insight-badge {
    box-shadow:0 0 15px rgba(16,185,129,0.2) !important;
    animation:badgeGlow 2.5s ease-in-out infinite;
}
@keyframes badgeGlow {
    0%,100% { box-shadow:0 0 15px rgba(16,185,129,0.15); }
    50%     { box-shadow:0 0 28px rgba(16,185,129,0.35); }
}

/* ─── MUSIC BAR PREMIUM ─── */
.music-bar {
    background:linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,41,59,0.85)) !important;
    border:1px solid rgba(37,99,235,0.25) !important;
    box-shadow:0 4px 16px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.04) !important;
}

/* ─── ANALYST INSIGHT PREMIUM ─── */
.analyst-insight {
    background:linear-gradient(135deg, rgba(16,185,129,0.08), rgba(16,185,129,0.03)) !important;
    border-left:4px solid #10b981 !important;
    box-shadow:0 4px 16px rgba(16,185,129,0.08) !important;
}

/* ─── WELCOME CARD PREMIUM ─── */
.welcome-card {
    background:linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,41,59,0.88)) !important;
    border:1px solid rgba(0,212,255,0.18) !important;
    box-shadow:0 12px 48px rgba(0,212,255,0.08) !important;
    backdrop-filter:blur(20px) !important;
}

/* ─── SHEET BADGES GLOW ─── */
.sheet-master {
    box-shadow:0 4px 12px rgba(245,158,11,0.15) !important;
}
.sheet-sub {
    box-shadow:0 4px 12px rgba(37,99,235,0.1) !important;
}

/* ─── EXPANDER PREMIUM ─── */
.streamlit-expanderHeader {
    background:linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,41,59,0.8)) !important;
    border:1px solid rgba(0,212,255,0.15) !important;
    box-shadow:0 4px 16px rgba(0,0,0,0.2) !important;
}

/* ─── DIVIDERS GRADIENT ─── */
hr {
    border:none !important;
    height:1px !important;
    background:linear-gradient(90deg, transparent, rgba(0,212,255,0.25), transparent) !important;
    margin:12px 0 !important;
}

/* ─── SMOOTH ALL TRANSITIONS ─── */
.score-card, .bento-item, .metric-card,
.chat-user-bubble, .chat-assistant-bubble,
.stButton>button, .timeline-content {
    transition:all 0.3s cubic-bezier(0.16,1,0.3,1) !important;
}

/* ─── SCROLLBAR PREMIUM ─── */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:rgba(15,23,42,0.3); }
::-webkit-scrollbar-thumb {
    background:linear-gradient(180deg, #2563eb, #667eea);
    border-radius:3px;
}
::-webkit-scrollbar-thumb:hover {
    background:linear-gradient(180deg, #00d4ff, #2563eb);
}

/* ─── FLOATING ANIMATION ─── */
@keyframes floatingMotion {
    0%,100% { transform:translateY(0px); }
    50%     { transform:translateY(-8px); }
}
.floating { animation:floatingMotion 4s ease-in-out infinite; }

/* ─── SAVE PROMPT PREMIUM ─── */
.save-prompt {
    background:linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.88)) !important;
    border:1px solid rgba(245,158,11,0.3) !important;
    box-shadow:0 8px 32px rgba(245,158,11,0.08) !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LANDING PAGE — Premium Intro Experience
# ============================================================
# LANDING PAGE — Premium Intro Experience
# LANDING PAGE — Premium Intro Experience
if "landing_shown" not in st.session_state:
    st.session_state.landing_shown = False

if not st.session_state.landing_shown:

    landing_html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>

    *{
        margin:0;
        padding:0;
        box-sizing:border-box;
        font-family:Inter,sans-serif;
    }

    body{
        overflow:hidden;
        background:#050816;
    }

    .container{
        width:100vw;
        height:100vh;
        position:relative;
        background:
        radial-gradient(circle at 50% 50%, rgba(0,180,255,.18), transparent 45%),
        radial-gradient(circle at 20% 20%, rgba(145,0,255,.12), transparent 30%),
        #050816;
        overflow:hidden;
    }

    .particle{
        position:absolute;
        width:4px;
        height:4px;
        border-radius:50%;
        background:white;
        opacity:.5;
        animation:float 12s linear infinite;
    }

    @keyframes float{
        from{
            transform:translateY(100vh);
        }
        to{
            transform:translateY(-100vh);
        }
    }

    .slide{
        position:absolute;
        inset:0;
        display:flex;
        justify-content:center;
        align-items:center;
        flex-direction:column;
        color:white;
        opacity:0;
        transition:all 1.5s ease;
    }

    .active{
        opacity:1;
    }

    .brain{
        width:280px;
        height:280px;
        border-radius:50%;
        background:
        radial-gradient(circle,
        rgba(0,255,255,.9),
        rgba(0,120,255,.2),
        transparent);
        box-shadow:
        0 0 80px cyan,
        0 0 150px #00bfff;
        animation:pulse 3s infinite;
    }

    @keyframes pulse{
        0%,100%{transform:scale(1);}
        50%{transform:scale(1.08);}
    }

    .title{
        font-size:70px;
        font-weight:800;
        margin-top:40px;
        text-align:center;
        letter-spacing:2px;
    }

    .subtitle{
        font-size:22px;
        color:#a5d8ff;
        margin-top:10px;
    }

    .glass{
        backdrop-filter:blur(20px);
        background:rgba(255,255,255,.05);
        border:1px solid rgba(255,255,255,.1);
        border-radius:24px;
        padding:20px;
        min-width:220px;
        text-align:center;
    }

    .kpi-wrap{
        display:flex;
        gap:20px;
        margin-top:40px;
        flex-wrap:wrap;
        justify-content:center;
    }

    .kpi{
        backdrop-filter:blur(20px);
        background:rgba(255,255,255,.05);
        border:1px solid rgba(255,255,255,.1);
        border-radius:20px;
        padding:20px 30px;
        min-width:180px;
    }

    .kpi h2{
        color:#00e5ff;
    }

    .enter{
        padding:18px 40px;
        border:none;
        border-radius:40px;
        background:linear-gradient(90deg,#00d4ff,#8a2be2);
        color:white;
        font-size:18px;
        cursor:pointer;
        margin-top:40px;
        box-shadow:0 0 40px rgba(0,200,255,.5);
    }

    </style>
    </head>

    <body>

    <div class="container">

        <div id="slide1" class="slide active">
            <div class="brain"></div>

            <div class="title">
                HOME CARE AI OS
            </div>

            <div class="subtitle">
                Executive Intelligence Platform
            </div>

            <div class="kpi-wrap">
                <div class="glass">Inventory Intelligence</div>
                <div class="glass">Business Intelligence</div>
                <div class="glass">Forecasting</div>
                <div class="glass">Automation</div>
                <div class="glass">AI Agents</div>
            </div>
        </div>

        <div id="slide2" class="slide">
            <div class="brain"></div>

            <div class="title">
                INITIALIZING
            </div>

            <div class="subtitle">
                AI SYSTEMS...
            </div>
        </div>

        <div id="slide3" class="slide">

            <div class="title">
                Executive Dashboard
            </div>

            <div class="kpi-wrap">

                <div class="kpi">
                    <small>Inventory</small>
                    <h2>98%</h2>
                </div>

                <div class="kpi">
                    <small>Revenue</small>
                    <h2>+23%</h2>
                </div>

                <div class="kpi">
                    <small>Forecast</small>
                    <h2>+18%</h2>
                </div>

                <div class="kpi">
                    <small>AI Agents</small>
                    <h2>ONLINE</h2>
                </div>

            </div>

            <button class="enter">
                ENTER PLATFORM
            </button>

        </div>

    </div>

    <script>

    for(let i=0;i<100;i++){

        let p=document.createElement("div");

        p.className="particle";

        p.style.left=Math.random()*100+"vw";

        p.style.animationDuration=
        (8+Math.random()*10)+"s";

        document.querySelector(".container")
        .appendChild(p);
    }

    setTimeout(()=>{
        document.getElementById("slide1").classList.remove("active");
        document.getElementById("slide2").classList.add("active");
    },5000);

    setTimeout(()=>{
        document.getElementById("slide2").classList.remove("active");
        document.getElementById("slide3").classList.add("active");
    },10000);

    </script>

    </body>
    </html>
    """

    components.html(
        landing_html,
        height=1000,
        scrolling=False
    )

    import time
    time.sleep(15)

    st.session_state.landing_shown = True
    st.rerun()


# ============================================================
# SIDEBAR
# ============================================================
sb = st.sidebar

sb.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;padding:14px 4px 18px 4px;">
    {logo_html}
    <div>
        <div style="font-size:1rem;font-weight:900;
             background:linear-gradient(135deg,#dc2626,#f97316);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;line-height:1.1;">Home Care AI OS</div>
        <div style="font-size:0.67rem;color:#475569;margin-top:2px;">Dual Agent Intelligence</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not GOOGLE_API_KEY:
    sb.error("⚠️ Google API Key مش موجود")
elif model_available:
    sb.success(f"✅ {MODEL_NAME}")
else:
    sb.warning(f"⚠️ {model_error or 'AI غير متاح'}")

if TELEGRAM_BOTS:
    sb.success(f"✅ Telegram: {len(TELEGRAM_BOTS)} بوت")
else:
    sb.info("ℹ️ Telegram غير مفعّل")

sb.divider()

# ── ADMIN ──
current_user_data = get_user(st.session_state.current_user)
is_admin = bool(current_user_data and current_user_data['role'] == 'admin')
if current_user_data: update_user_login(st.session_state.current_user)

if is_admin:
    with sb.expander("👑 لوحة الإدارة"):
        nu = st.text_input("اسم مستخدم جديد", key="nu")
        nr = st.selectbox("الدور", ["user","manager","analyst"], key="nr")
        ne = st.text_input("الإيميل", key="ne")
        if st.button("➕ إضافة", key="add_user"):
            if nu and create_user(nu, nr, ne):
                st.success(f"✅ {nu}")
                st.rerun()
        st.divider()
        for u in list_all_users():
            c1, c2 = st.columns([3,1])
            with c1: st.caption(f"👤 {u[0]} [{u[1]}]")
            with c2:
                if st.button("↩", key=f"sw_{u[0]}"):
                    st.session_state.current_user = u[0]
                    st.rerun()

    pending = get_pending_insights()
    if pending:
        with sb.expander(f"🧠 دروس جديدة ({len(pending)})"):
            for ins in pending:
                st.markdown(f"<div class='insight-badge'>💡 {ins[1]}</div>", unsafe_allow_html=True)
                if st.button("✅", key=f"ap_{ins[0]}"):
                    approve_insight(ins[0], st.session_state.current_user)
                    st.rerun()

sb.divider()

# ── FILE UPLOAD ──
sb.markdown("**📁 رفع الملفات**")
uploaded_file = sb.file_uploader("Excel / CSV", type=["xlsx","xlsm","csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith((".xlsx",".xlsm")):
            with st.spinner("🔍 بيحلل الشيتات..."):
                sheets_dict = read_all_sheets(uploaded_file)
            if sheets_dict:
                st.session_state.sheets_dict = sheets_dict
                st.session_state.df = sheets_dict[list(sheets_dict.keys())[0]]
                sb.success(f"✅ {uploaded_file.name} ({len(sheets_dict)} شيت)")
                master = sb.selectbox("🧠 الشيت الأساسي", list(sheets_dict.keys()))
                st.session_state.master_sheet = master
                remaining = [s for s in sheets_dict.keys() if s != master]
                subs = sb.multiselect("📊 الشيتات الفرعية", remaining,
                                       default=remaining[:5] if len(remaining)>5 else remaining)
                st.session_state.sub_sheets = subs
                save_file_metadata(
                    uploaded_file.name, master,
                    str(REPORTS_DIR / uploaded_file.name),
                    list(sheets_dict.keys()),
                    f"Master:{master}",
                    json.dumps(find_sheet_relationships(sheets_dict), ensure_ascii=False)
                )
        else:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.sheets_dict = {uploaded_file.name.replace('.csv',''): df}
            st.session_state.master_sheet = uploaded_file.name.replace('.csv','')
            sb.success(f"✅ {uploaded_file.name}")
    except Exception as e:
        sb.error(f"❌ {e}")

sb.divider()
st.session_state.show_charts = sb.toggle("📊 الرسوم البيانية", value=st.session_state.show_charts)
st.session_state.show_memory_timeline = sb.toggle("🕐 Memory Timeline", value=st.session_state.show_memory_timeline)

with sb.expander("📜 سجل الملفات"):
    for row in get_files_history(5):
        sb.caption(f"📄 {row[1]}\n🧠 {row[2]}\n⏰ {row[3][:10]}")

# ============================================================
# MAIN — HEADER
# ============================================================
role_badge = "ADMIN ⭐" if is_admin else (current_user_data['role'].upper() if current_user_data else "GUEST")

st.markdown(f"""
<div class="hc-header">
    {logo_html}
    <div>
        <div class="hc-title">🧠 Home Care AI OS</div>
        <p class="hc-sub">
            👤 {st.session_state.current_user} &nbsp;|&nbsp; 🎖️ {role_badge}
            &nbsp;|&nbsp; 🤖 Dual Agent &nbsp;|&nbsp; 📡 {MODEL_NAME}
            &nbsp;|&nbsp; 📁 {OUTPUT_DIR.name}
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SMART SCORES SECTION
# ============================================================
if st.session_state.df is not None:
    df = st.session_state.df
    
    inv   = calculate_inventory_health_score(df)
    cust  = calculate_customer_risk_score(df)
    cash  = calculate_cashflow_health(df)
    pred  = calculate_stock_prediction_score(df)
    ai_cf = calculate_ai_confidence(
        len(st.session_state.last_prompt),
        len(df),
        len(st.session_state.last_response or "")
    )
    
    scores = [
        {"label":"🏭 صحة المخزون",  "score": inv["score"],  "color": inv["color"],  "badge": inv["label"]},
        {"label":"👥 مخاطر العملاء","score": cust["score"], "color": cust["color"], "badge": cust["label"]},
        {"label":"💰 صحة التدفق",   "score": cash["score"], "color": cash["color"], "badge": cash["label"]},
        {"label":"🔮 تنبؤ المخزون", "score": pred["score"], "color": pred["color"], "badge": pred["label"]},
        {"label":"🤖 ثقة الـ AI",   "score": ai_cf["score"],"color": ai_cf["color"],"badge": ai_cf["label"]},
    ]
    
    cols = st.columns(5)
    for i, s in enumerate(scores):
        w = int(s['score'])
        r = s["color"][1:3]; g = s["color"][3:5]; b_c = s["color"][5:7]
        rr = int(r,16); gg = int(g,16); bb = int(b_c,16)
        with cols[i]:
            st.markdown(f"""
            <div class="score-card" style="--score-color:{s['color']};--score-rgb:{rr},{gg},{bb};">
                <div class="score-label">{s['label']}</div>
                <div class="score-value">{w}<span style="font-size:1rem;font-weight:400;">%</span></div>
                <span class="score-badge">{s['badge']}</span>
                <div class="score-bar">
                    <div class="score-bar-fill" style="width:{w}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# KPI METRICS + CHARTS
# ============================================================
if st.session_state.sheets_dict and st.session_state.master_sheet:
    mdf = st.session_state.sheets_dict[st.session_state.master_sheet]
    
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:1.8rem;">📊</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:3px;">الصفوف</div>
            <div style="font-size:1.7rem;font-weight:900;color:#00d4ff;margin-top:6px;">{len(mdf):,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:1.8rem;">📋</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:3px;">الأعمدة</div>
            <div style="font-size:1.7rem;font-weight:900;color:#667eea;margin-top:6px;">{len(mdf.columns)}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        missing = int(mdf.isna().sum().sum())
        m_color = "#ef4444" if missing > 0 else "#10b981"
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:1.8rem;">🔍</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:3px;">فارغة</div>
            <div style="font-size:1.7rem;font-weight:900;color:{m_color};margin-top:6px;">{missing}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:1.8rem;">🔢</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:3px;">أرقام</div>
            <div style="font-size:1.7rem;font-weight:900;color:#10b981;margin-top:6px;">{len(mdf.select_dtypes(include='number').columns)}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:1.8rem;">📑</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:3px;">الشيتات</div>
            <div style="font-size:1.7rem;font-weight:900;color:#f59e0b;margin-top:6px;">{len(st.session_state.sheets_dict)}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    
    if st.session_state.show_charts:
        kpi_fig = generate_kpi_dashboard(st.session_state.sheets_dict, st.session_state.master_sheet)
        if kpi_fig:
            st.plotly_chart(kpi_fig, use_container_width=True, config={"displayModeBar": False})
    
    with st.expander("📊 معاينة البيانات — هيكل + علاقات", expanded=False):
        t1, t2, t3 = st.tabs(["📋 البيانات", "🗂️ الشيتات", "🔗 العلاقات"])
        with t1:
            st.dataframe(mdf.head(10), use_container_width=True)
        with t2:
            for sn, sdf in st.session_state.sheets_dict.items():
                stype = detect_sheet_type(sn, sdf)
                badge = "sheet-master" if sn == st.session_state.master_sheet else "sheet-sub"
                icon  = "⭐" if sn == st.session_state.master_sheet else "📋"
                st.markdown(
                    f"<span class='sheet-badge {badge}'>{icon} {sn}</span> "
                    f"<span style='color:#64748b;font-size:0.75rem;'>{stype} | {len(sdf)} صف</span>",
                    unsafe_allow_html=True)
        with t3:
            rels = find_sheet_relationships(st.session_state.sheets_dict)
            if rels:
                for sheet, srels in rels.items():
                    if srels:
                        top = srels[0]
                        st.markdown(
                            f"<span style='color:#60a5fa;font-weight:700;'>{sheet}</span>"
                            f"<span style='color:#64748b;'> ↔ {top['sheet']}: "
                            f"<code style='color:#a5b4fc;'>{', '.join(top['common_columns'][:4])}</code></span>",
                            unsafe_allow_html=True)
            else:
                st.info("لم تُكتشف علاقات مباشرة")

# ============================================================
# AI MEMORY TIMELINE
# ============================================================
if st.session_state.show_memory_timeline:
    st.markdown("---")
    st.markdown("<div class='chat-section-title'>🕐 AI Memory Timeline</div>", unsafe_allow_html=True)
    
    timeline_items = load_memory_timeline(st.session_state.current_user, limit=30)
    
    if timeline_items:
        st.markdown("<div class='timeline-container'>", unsafe_allow_html=True)
        st.markdown("<div class='timeline-line'></div>", unsafe_allow_html=True)
        
        for item in reversed(timeline_items[-15:]):
            dot_cls = "user" if item["role"] == "user" else ""
            preview = item["content"][:120] + "..." if len(item["content"]) > 120 else item["content"]
            intent_html = (f"<span class='timeline-intent'>{item['intent']}</span>"
                           if item.get("intent") and item["intent"] != "عام" else "")
            ts = item["created_at"][:16] if item.get("created_at") else ""
            role_label = "👤 أنت" if item["role"] == "user" else "🤖 AI"
            
            st.markdown(f"""
            <div class="timeline-item">
                <div class="timeline-dot {dot_cls}"></div>
                <div class="timeline-content">
                    <div class="timeline-meta">{role_label} &nbsp;|&nbsp; {ts}</div>
                    <div class="timeline-text">{preview}</div>
                    {intent_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("لا توجد محادثات سابقة في الذاكرة")

# ============================================================
# CHAT SECTION
# ============================================================
st.markdown("<div class='chat-section-title'>💬 محادثة مع Home Care AI</div>", unsafe_allow_html=True)
st.markdown(
    "<span style='color:#475569;font-size:0.76rem;'>"
    "🤖 مهاب + وكيل التحليل — يفهم كل أنواع الكلام المصري والعربي والإنجليزي"
    "</span>", unsafe_allow_html=True)

# Welcome
if not st.session_state.messages:
    st.markdown("""
    <div class='welcome-card'>
        <div style='font-size:2rem;margin-bottom:12px;'>👋</div>
        <h2>أهلاً! أنا مهاب — Home Care AI</h2>
        <p>
            🤖 <b>Dual Agent</b> — وكيلان يناقشان لإجابة أعمق وأدق<br>
            📊 <b>رسوم بيانية احترافية</b> تلقائياً مع كل تحليل<br>
            🏭 <b>Smart Scores</b> — صحة المخزون، مخاطر العملاء، التنبؤ<br>
            🕐 <b>Memory Timeline</b> — تاريخ كامل للمحادثات<br>
            📁 ارفع ملفك وابدأ — أو اسألني أي حاجة
        </p>
    </div>
    """, unsafe_allow_html=True)

# عرض المحادثات
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-user-wrapper">
            <div>
                <div class="chat-user-label">أنت 👤</div>
                <div class="chat-user-bubble">{msg['content']}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-assistant-wrapper">
            <div style="width:100%;max-width:78%;">
                <div class="chat-assistant-label">
                    🏠 Home Care AI
                    <span class="agent-tag">مهاب</span>
                    <span class="analyst-tag">وكيل التحليل</span>
                </div>
                <div class="chat-assistant-bubble">{msg['content']}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        if st.session_state.show_charts and msg.get("charts"):
            cols = st.columns(min(len(msg["charts"]), 2))
            for ci, fig in enumerate(msg["charts"]):
                with cols[ci % 2]:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        if msg.get("analyst"):
            st.markdown(
                f"<div class='analyst-insight'>🔬 <b>وكيل التحليل:</b><br>"
                f"{msg['analyst'][:500]}{'...' if len(msg.get('analyst',''))>500 else ''}</div>",
                unsafe_allow_html=True)

# ============================================================
# PENDING TELEGRAM SEND
# ============================================================
if st.session_state.pending_send_telegram:
    st.markdown("---")
    st.markdown("### 📲 إرسال على تيليجرام")
    
    if len(TELEGRAM_BOTS) > 1:
        chosen_bot   = st.selectbox("اختار البوت:", list(TELEGRAM_BOTS.keys()))
        chosen_token = TELEGRAM_BOTS[chosen_bot]
    elif TELEGRAM_BOTS:
        chosen_bot   = list(TELEGRAM_BOTS.keys())[0]
        chosen_token = list(TELEGRAM_BOTS.values())[0]
        st.info(f"البوت: {chosen_bot}")
    else:
        chosen_token = None
    
    tg_chat_id = st.text_input("Chat ID:", value=ADMIN_CHAT_ID)
    
    tc1, tc2 = st.columns(2)
    with tc1:
        if st.button("✅ إرسال"):
            if chosen_token and tg_chat_id:
                ok = send_telegram_message(chosen_token, tg_chat_id, st.session_state.pending_send_telegram)
                st.success("✅ تم الإرسال!") if ok else st.error("❌ فشل — تأكد من التوكن والـ Chat ID")
            st.session_state.pending_send_telegram = None
            st.rerun()
    with tc2:
        if st.button("❌ إلغاء"):
            st.session_state.pending_send_telegram = None
            st.rerun()

# ============================================================
# SAVE REPORT PROMPT
# ============================================================
if st.session_state.ask_save_report and st.session_state.last_response:
    st.markdown("---")
    st.markdown("<div class='save-prompt'>", unsafe_allow_html=True)
    st.markdown("<div class='save-title'>💾 هل تريد حفظ هذا التحليل؟</div>", unsafe_allow_html=True)
    st.caption(f"📁 {OUTPUT_DIR}")
    
    spc1, spc2, spc3, spc4 = st.columns(4)
    with spc1:
        if st.button("📋 PDF"):
            fp = export_to_pdf("تحليل Home Care", st.session_state.last_response, st.session_state.current_user)
            if fp:
                try:
                    import shutil; shutil.copy(str(fp), str(OUTPUT_DIR / fp.name))
                    st.success(f"✅ {fp.name}")
                except: st.success(f"✅ {fp}")
            st.session_state.ask_save_report = False
            st.rerun()
    with spc2:
        if st.button("📊 Excel"):
            dd = {"التحليل": pd.DataFrame([{"النتيجة": st.session_state.last_response}])}
            if st.session_state.sheets_dict:
                for sn, sdf in list(st.session_state.sheets_dict.items())[:5]:
                    dd[sn] = sdf
            fp = export_to_excel("تحليل Home Care", dd, st.session_state.current_user)
            if fp:
                try:
                    import shutil; shutil.copy(str(fp), str(OUTPUT_DIR / fp.name))
                except: pass
                st.success(f"✅ {fp.name}")
            st.session_state.ask_save_report = False
            st.rerun()
    with spc3:
        if st.button("📝 Markdown"):
            fn = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            (OUTPUT_DIR / fn).write_text(
                f"# {st.session_state.last_prompt}\n\n{st.session_state.last_response}", encoding="utf-8")
            st.success(f"✅ {fn}")
            st.session_state.ask_save_report = False
            st.rerun()
    with spc4:
        if st.button("⏭️ تخطي"):
            st.session_state.ask_save_report = False
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# CHAT INPUT
# ============================================================
user_input = st.chat_input("اكتب طلبك هنا — بأي أسلوب يعجبك 😊")

if user_input:
    st.session_state.messages.append({"role":"user","content":user_input,"charts":[],"analyst":""})
    st.session_state.last_prompt = user_input
    st.session_state.ask_save_report = False
    
    intents_check = detect_intent(user_input)
    
    # Telegram intent
    if "تليجرام" in intents_check:
        last_resp = st.session_state.last_response or user_input
        st.session_state.pending_send_telegram = last_resp
        st.session_state.messages.append({
            "role": "assistant",
            "content": "📲 تمام! جاهز أبعت — اختار البوت وأدخل الـ Chat ID.",
            "charts": [], "analyst": ""
        })
        st.rerun()
    
    # Build context
    master_context = ""
    data_context   = ""
    has_data = bool(st.session_state.sheets_dict and st.session_state.master_sheet)
    
    if has_data:
        master_context = build_deep_sheet_context(
            st.session_state.sheets_dict,
            st.session_state.master_sheet,
            st.session_state.sub_sheets)
        mdf = st.session_state.sheets_dict[st.session_state.master_sheet]
        data_context = f"الشيت الأساسي: {st.session_state.master_sheet}\n"
        data_context += f"الصفوف: {len(mdf)} | الأعمدة: {list(mdf.columns)}\n"
        for col in mdf.select_dtypes(include='number').columns[:6]:
            try: data_context += f"\n{col}: Σ={mdf[col].sum():.2f} | μ={mdf[col].mean():.2f} | max={mdf[col].max():.2f}"
            except: pass
        for sub in st.session_state.sub_sheets[:4]:
            if sub in st.session_state.sheets_dict:
                sdf = st.session_state.sheets_dict[sub]
                data_context += f"\n\n{sub}: {len(sdf)} صف | {list(sdf.columns[:8])}"
    elif st.session_state.df is not None:
        df = st.session_state.df
        data_context = f"الصفوف:{len(df)} | الأعمدة:{list(df.columns)}\n{df.head(2).to_string()}"
    
    memory_items = load_memory(st.session_state.current_user, limit=15)
    memory_text  = "\n".join(f"{m['role']}: {m['content']}" for m in memory_items)
    user_intel   = get_user_intelligence(st.session_state.current_user)
    
    # AI Thinking animation placeholder
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown("""
    <div class="chat-assistant-wrapper">
        <div style="width:100%;max-width:78%;">
            <div class="chat-assistant-label">🏠 Home Care AI <span class="agent-tag">مهاب</span> <span class="analyst-tag">وكيل التحليل</span></div>
            <div class="ai-thinking-stream">
                <div class="thinking-dots">
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                </div>
                <span class="thinking-text" style="color:#64748b;font-size:0.82rem;">مهاب بيفكر ويحلل...</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ai_response, analyst_insight = dual_agent_response(
        user_input, data_context, memory_text,
        st.session_state.current_user, master_context, user_intel)
    
    thinking_placeholder.empty()
    
    # Charts
    charts = []
    is_analysis = any(i in intents_check for i in ["تحليل","مخزون","مبيعات","kpi","مقارنة","كود","عميل","مالي"])
    if st.session_state.show_charts and has_data and is_analysis:
        active_df = st.session_state.sheets_dict.get(st.session_state.master_sheet, st.session_state.df)
        with st.spinner("📊 رسم المخططات..."):
            charts = auto_generate_charts(active_df, st.session_state.master_sheet or "Data", user_input)
    
    st.session_state.last_response  = ai_response
    st.session_state.last_analyst   = analyst_insight
    st.session_state.last_charts    = charts
    st.session_state.messages.append({
        "role": "assistant", "content": ai_response,
        "charts": charts, "analyst": analyst_insight
    })
    
    save_memory(st.session_state.current_user, "web", "user", user_input, ", ".join(intents_check))
    save_memory(st.session_state.current_user, "web", "assistant", ai_response)
    
    style = "فكاهي" if "فكاهي" in intents_check else "تحليلي" if "تحليل" in intents_check else "عام"
    update_user_intelligence(st.session_state.current_user, style,
                              ", ".join(intents_check), f"يسأل عن: {', '.join(intents_check)}")
    
    if is_analysis and has_data:
        st.session_state.ask_save_report = True
    
    st.rerun()

# ============================================================
# EXTRA ACTIONS
# ============================================================
if st.session_state.last_response and not st.session_state.ask_save_report:
    st.markdown("---")
    ea1, ea2, ea3, ea4, ea5 = st.columns(5)
    
    with ea1:
        if st.button("📋 PDF", key="ea_pdf"):
            fp = export_to_pdf("تحليل Home Care", st.session_state.last_response, st.session_state.current_user)
            if fp:
                try:
                    import shutil; shutil.copy(str(fp), str(OUTPUT_DIR / fp.name))
                except: pass
                st.success(f"✅ {fp.name}")
    with ea2:
        if st.button("📊 Excel", key="ea_xl"):
            dd = {"التحليل": pd.DataFrame([{"النتيجة": st.session_state.last_response}])}
            if st.session_state.sheets_dict:
                for sn, sdf in list(st.session_state.sheets_dict.items())[:5]: dd[sn] = sdf
            fp = export_to_excel("تحليل Home Care", dd, st.session_state.current_user)
            if fp:
                try:
                    import shutil; shutil.copy(str(fp), str(OUTPUT_DIR / fp.name))
                except: pass
                st.success(f"✅ {fp.name}")
    with ea3:
        if st.button("📲 تليجرام", key="ea_tg"):
            st.session_state.pending_send_telegram = st.session_state.last_response
            st.rerun()
    with ea4:
        if st.button("📝 Markdown", key="ea_md"):
            fn = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            (OUTPUT_DIR / fn).write_text(
                f"# {st.session_state.last_prompt}\n\n{st.session_state.last_response}", encoding="utf-8")
            st.success(f"✅ {fn}")
    with ea5:
        if st.button("🗑️ مسح", key="ea_clr"):
            st.session_state.messages = []
            st.session_state.last_response = None
            st.session_state.ask_save_report = False
            st.rerun()

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#1e293b;font-size:0.7rem;padding:6px;'>"
    f"🏠 Home Care AI OS &nbsp;•&nbsp; Dual Agent System &nbsp;•&nbsp; "
    f"Smart Scores &nbsp;•&nbsp; Memory Timeline &nbsp;•&nbsp; {MODEL_NAME} &nbsp;•&nbsp; "
    f"📁 {OUTPUT_DIR}"
    f"</div>",
    unsafe_allow_html=True)