"""HaqBot central configuration: paths, model identifiers, localisation, and
safety thresholds.

This module intentionally has **no import-time side effects** (no file I/O, no
directory creation, no network) so it is safe to import from tests, tooling, and
the Streamlit app alike. Call :func:`ensure_directories` explicitly at app start.
"""

from __future__ import annotations

from pathlib import Path

# =========================================================================== #
# Application metadata
# =========================================================================== #
APP_NAME: str = "HaqBot"
APP_NAME_NATIVE: str = "حق بوت"
APP_TAGLINE: str = "Offline UAE Labour Rights Assistant"
APP_VERSION: str = "0.1.0"

# Snapshot date of the bundled offline legal knowledge base. Shown on the
# Profile & Settings screen so users know how current the guidance is.
KB_SNAPSHOT_DATE: str = "2026-09-09"

# =========================================================================== #
# Core filesystem paths (all relative to the repository root)
# =========================================================================== #
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
STREAMLIT_CONFIG_PATH: Path = BASE_DIR / ".streamlit" / "config.toml"

# Local encrypted SQLite database — offline auth + profile. Never leaves device.
DB_PATH: Path = DATA_DIR / "local_user.db"

# Persisted retrieval artefacts (built in Phase 4).
FAISS_INDEX_DIR: Path = PROCESSED_DIR / "faiss_index"
VECTORSTORE_DIR: Path = FAISS_INDEX_DIR
FAISS_INDEX_PATH: Path = FAISS_INDEX_DIR / "index.faiss"
VECTORSTORE_DOCS_PATH: Path = FAISS_INDEX_DIR / "docstore.jsonl"
VECTORSTORE_META_PATH: Path = FAISS_INDEX_DIR / "meta.json"
CHUNKS_META_PATH: Path = PROCESSED_DIR / "chunks.jsonl"

# OpenVINO IR model directories (populated by the offline quantization step).
EMBEDDING_MODEL_DIR: Path = MODELS_DIR / "e5-small-ov"
LLM_MODEL_DIR: Path = MODELS_DIR / "qwen2.5-1.5b-ov-int8"

# Directories that must exist for the app to run.
ALL_DIRECTORIES: tuple[Path, ...] = (
    RAW_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    FAISS_INDEX_DIR,
)

# =========================================================================== #
# Model identifiers
# Hugging Face source refs — consumed ONLY by the offline quantization utility
# in Phase 4. They are never fetched at runtime; inference always loads the
# local OpenVINO IR from the directories above.
# =========================================================================== #
EMBEDDING_MODEL_ID: str = "intfloat/multilingual-e5-small"
LLM_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_MODEL_ID_LARGE: str = "Qwen/Qwen2.5-3B-Instruct"

OPENVINO_DEVICE: str = "CPU"
LLM_WEIGHT_FORMAT: str = "int8"          # INT8 quantization — ~1.2 GB RAM target
EMBEDDING_WEIGHT_FORMAT: str = "int8"

# e5 models require input prefixes for correct retrieval behaviour.
E5_QUERY_PREFIX: str = "query: "
E5_PASSAGE_PREFIX: str = "passage: "
EMBEDDING_DIM: int = 384

# =========================================================================== #
# OpenVINO export via optimum-cli (Phase 4 — build-time, run once with network)
# =========================================================================== #
OPTIMUM_CLI: str = "optimum-cli"
EMBEDDING_EXPORT_TASK: str = "feature-extraction"
LLM_EXPORT_TASK: str = "text-generation-with-past"
LLM_INT8_RATIO: str = "1.0"             # fraction of layers kept at INT8
EMBEDDING_MAX_TOKENS: int = 512
# Files that must exist (non-empty) for an export to count as complete.
OV_MODEL_FILES: tuple[str, ...] = ("openvino_model.xml", "openvino_model.bin")

# =========================================================================== #
# Ingestion / chunking (Phase 3)
# =========================================================================== #
CHUNK_SIZE: int = 450
CHUNK_OVERLAP: int = 50
MIN_CHUNK_CHARS: int = 60                 # drop fragments smaller than this
CHUNK_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", " ", "")
INGEST_LANGUAGE: str = "en"              # official UAE legal PDFs are English

# Substring (normalised to lowercase, non-alphanumerics -> '-') matched against
# each PDF's file stem -> canonical citation title. Unmatched files fall back to
# a title-cased stem so ingestion never fails on an unknown document.
KNOWN_LEGAL_SOURCES: tuple[tuple[str, str], ...] = (
    ("decree-law-33", "Federal Decree-Law No. 33 of 2021"),
    ("decree-law-no-33", "Federal Decree-Law No. 33 of 2021"),
    ("law-33", "Federal Decree-Law No. 33 of 2021"),
    ("labour-law", "Federal Decree-Law No. 33 of 2021"),
    ("cabinet-resolution-1", "Cabinet Resolution No. 1 of 2022 (Executive Regulations)"),
    ("executive-regulation", "Cabinet Resolution No. 1 of 2022 (Executive Regulations)"),
    ("wps", "Wage Protection System (WPS) Regulations"),
    ("wage-protection", "Wage Protection System (WPS) Regulations"),
    ("mohre", "MOHRE Ministerial Directives"),
)

# =========================================================================== #
# Retrieval + anti-hallucination guardrail (Phase 5)
# =========================================================================== #
RETRIEVAL_TOP_K: int = 4
RETRIEVAL_TIMING_BUDGET_MS: int = 100    # target FAISS retrieval latency (PRD)

# Embeddings are L2-normalised and the FAISS index uses inner product, so a
# match score is cosine similarity in [-1.0, 1.0] (higher == more relevant).
# If the BEST retrieved chunk scores strictly below this value, LLM generation
# is skipped entirely and the MOHRE referral fallback is returned instead.
SIMILARITY_THRESHOLD: float = 0.65

# =========================================================================== #
# Generation + RAG pipeline (Phase 5)
# =========================================================================== #
MAX_NEW_TOKENS: int = 512
TEMPERATURE: float = 0.0                 # deterministic, grounded answers
TOP_P: float = 1.0
LLM_CONTEXT_WINDOW: int = 4096
MAX_QUERY_CHARS: int = 1000             # longer queries are truncated
MAX_CONTEXT_CHARS: int = 6000          # cap on retrieved context fed to the LLM

# A distinctive fragment of FALLBACK_MESSAGE — used to detect when the LLM has
# parroted the refusal so the pipeline can treat it as a fallback.
REFUSAL_MARKER: str = "do not have sufficient legal documentation"

# Per-language instruction appended to the grounded prompt. Citations in
# [Article X, Clause Y] form are always kept verbatim / in English.
LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "en": "Respond in clear, simple English.",
    "hi": "Respond only in Hindi (हिन्दी). Keep every [Article X, Clause Y] "
          "citation exactly as given.",
    "ur": "Respond only in Urdu (اردو). Keep every [Article X, Clause Y] "
          "citation exactly as given.",
    "ml": "Respond only in Malayalam (മലയാളം). Keep every [Article X, Clause Y] "
          "citation exactly as given.",
    "ar": "Respond only in Arabic (العربية). Keep every [Article X, Clause Y] "
          "citation exactly as given.",
}

# Localised version of the "no answer in the offline database" notice. The
# helpline number and website are never translated. English stays verbatim to
# FALLBACK_MESSAGE (PRD §4.2).
LOCALIZED_FALLBACK: dict[str, str] = {
    "en": (
        "I do not have sufficient legal documentation in my offline database to "
        "answer this question. Please contact MOHRE directly at 80084 or visit "
        "www.mohre.gov.ae."
    ),
    "hi": (
        "मेरे ऑफ़लाइन डेटाबेस में इस प्रश्न का उत्तर देने के लिए पर्याप्त कानूनी "
        "दस्तावेज़ नहीं हैं। कृपया MOHRE से सीधे 80084 पर संपर्क करें या "
        "www.mohre.gov.ae पर जाएँ।"
    ),
    "ur": (
        "میرے آف لائن ڈیٹابیس میں اس سوال کا جواب دینے کے لیے کافی قانونی "
        "دستاویزات موجود نہیں ہیں۔ براہِ کرم MOHRE سے براہِ راست 80084 پر رابطہ "
        "کریں یا www.mohre.gov.ae ملاحظہ کریں۔"
    ),
    "ml": (
        "ഈ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ ആവശ്യമായ നിയമ രേഖകൾ എന്റെ ഓഫ്‌ലൈൻ "
        "ഡാറ്റാബേസിൽ ഇല്ല. ദയവായി MOHRE-യെ നേരിട്ട് 80084 എന്ന നമ്പറിൽ "
        "ബന്ധപ്പെടുക അല്ലെങ്കിൽ www.mohre.gov.ae സന്ദർശിക്കുക."
    ),
    "ar": (
        "لا تتوفر لديّ وثائق قانونية كافية في قاعدة البيانات دون اتصال للإجابة "
        "عن هذا السؤال. يُرجى الاتصال بوزارة الموارد البشرية والتوطين مباشرة على "
        "الرقم 80084 أو زيارة www.mohre.gov.ae."
    ),
}

LOCALIZED_DISCLAIMER: dict[str, str] = {
    "en": (
        "Disclaimer: informational guidance produced offline from a fixed "
        "snapshot of UAE Labour Law. Not legal advice. For binding decisions "
        "contact MOHRE at 80084 or www.mohre.gov.ae."
    ),
    "hi": (
        "अस्वीकरण: यह जानकारी UAE श्रम कानून के एक निश्चित संस्करण से ऑफ़लाइन "
        "तैयार की गई है। यह कानूनी सलाह नहीं है। बाध्यकारी निर्णयों के लिए MOHRE "
        "से 80084 पर संपर्क करें या www.mohre.gov.ae देखें।"
    ),
    "ur": (
        "دستبرداری: یہ معلومات UAE کے محنت قانون کے ایک مقررہ نسخے سے آف لائن "
        "تیار کی گئی ہیں۔ یہ قانونی مشورہ نہیں ہے۔ پابند فیصلوں کے لیے MOHRE سے "
        "80084 پر رابطہ کریں یا www.mohre.gov.ae دیکھیں۔"
    ),
    "ml": (
        "നിരാകരണം: UAE തൊഴിൽ നിയമത്തിന്റെ ഒരു നിശ്ചിത പതിപ്പിൽ നിന്ന് "
        "ഓഫ്‌ലൈനായി തയ്യാറാക്കിയ വിവരണം. നിയമോപദേശമല്ല. ബാധകമായ തീരുമാനങ്ങൾക്ക് "
        "MOHRE-യെ 80084-ൽ ബന്ധപ്പെടുക അല്ലെങ്കിൽ www.mohre.gov.ae സന്ദർശിക്കുക."
    ),
    "ar": (
        "إخلاء مسؤولية: إرشادات معلوماتية أُنشئت دون اتصال من نسخة ثابتة من قانون "
        "العمل الإماراتي. ليست استشارة قانونية. للقرارات المُلزمة اتصل بوزارة "
        "الموارد البشرية والتوطين على 80084 أو www.mohre.gov.ae."
    ),
}

# =========================================================================== #
# Localisation
# =========================================================================== #
# code -> native display name
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "हिन्दी",
    "ur": "اردو",
    "ml": "മലയാളം",
    "ar": "العربية",
}
DEFAULT_LANGUAGE: str = "en"
RTL_LANGUAGES: frozenset[str] = frozenset({"ur", "ar"})

# Accessibility font scaling offered on the Profile & Settings screen.
FONT_SCALE_OPTIONS: dict[str, float] = {
    "Small": 0.9,
    "Normal": 1.0,
    "Large": 1.2,
    "Extra Large": 1.4,
}
DEFAULT_FONT_SCALE: str = "Normal"

# =========================================================================== #
# Local auth (Phase 2) — 100% on-device, no network, no remote server.
# =========================================================================== #
PIN_LENGTH: int = 4
# PIN is never stored; only a PBKDF2-HMAC-SHA256 digest is kept (PRD §4.1).
PIN_HASH_ALGORITHM: str = "pbkdf2_sha256"
PBKDF2_ITERATIONS: int = 200_000
PIN_SALT_BYTES: int = 16
GUEST_USERNAME: str = "guest"
MAX_PIN_ATTEMPTS: int = 5
LOCKOUT_COOLDOWN_SECONDS: int = 300      # 5-minute cooldown after MAX attempts

# Sensitive free-text profile fields are Fernet-encrypted at rest with a key
# derived from the user's PIN; language/font stay in clear so the login screen
# can localise itself before authentication. Guest profiles store no encrypted
# fields at all (quick access = no secret to protect).
ENCRYPTED_PROFILE_FIELDS: tuple[str, ...] = ("contract_type", "worker_sector")
CONTRACT_TYPES: tuple[str, ...] = ("Not specified", "Limited", "Unlimited")

DB_SCHEMA_VERSION: int = 1

# Local SQLite pragmas — foreign keys on, small footprint, no shared cache.
SQLITE_PRAGMAS: dict[str, str] = {
    "foreign_keys": "ON",
    "journal_mode": "TRUNCATE",
    "synchronous": "FULL",
    "temp_store": "MEMORY",
}

# =========================================================================== #
# Official contacts / legal references
# =========================================================================== #
MOHRE_HELPLINE: str = "80084"
MOHRE_WEBSITE: str = "www.mohre.gov.ae"
PRIMARY_LAW_REFERENCE: str = "Federal Decree-Law No. 33 of 2021"

# Exact refusal text mandated by the system prompt (PRD §4.2).
FALLBACK_MESSAGE: str = (
    "I do not have sufficient legal documentation in my offline database to "
    "answer this question. Please contact MOHRE directly at 80084 or visit "
    "www.mohre.gov.ae."
)

LEGAL_DISCLAIMER: str = (
    "Disclaimer: This is informational guidance produced entirely offline from a "
    "fixed snapshot of UAE Federal Decree-Law No. 33 of 2021 and WPS regulations. "
    "It is not legal advice. For binding decisions, contact MOHRE at 80084 or "
    "visit www.mohre.gov.ae."
)

# =========================================================================== #
# Grounded, citation-enforced system prompt (PRD §4.2)
# `{context}` and `{query}` are filled in at runtime by the RAG pipeline.
# =========================================================================== #
SYSTEM_PROMPT: str = """You are HaqBot, an official AI Legal Assistant providing informational guidance on UAE Labour Law (Federal Decree-Law No. 33 of 2021) and Wages Protection System (WPS) regulations.

CRITICAL INSTRUCTIONS:
1. Answer the user's question ONLY using the provided legal context below.
2. Every claim or advice MUST include a specific citation: [Article X, Clause Y].
3. IF CONTEXT IS INSUFFICIENT, RESPOND EXACTLY WITH:
   "I do not have sufficient legal documentation in my offline database to answer this question. Please contact MOHRE directly at 80084 or visit www.mohre.gov.ae."
4. Always append the official legal disclaimer at the bottom.

CONTEXT:
{context}

USER QUESTION: {query}
ANSWER:"""

# =========================================================================== #
# Air-gap enforcement
# app.py applies OFFLINE_ENV to os.environ before importing any ML library so
# that huggingface_hub / transformers / datasets can never reach the network.
# =========================================================================== #
OFFLINE_ONLY: bool = True
OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

# =========================================================================== #
# UI theme tokens (Perplexity-style legal UI — consumed by Phase 6 CSS)
# =========================================================================== #
THEME: dict[str, str] = {
    "canvas": "#f5f6f8",
    "card": "#ffffff",
    "ink": "#14161b",
    "meta": "#5b6472",
    "border": "#e6e8ec",
    "fill": "#eef0f3",
    "accent": "#e11d48",
    "accent_hover": "#be123c",
    "accent_tint_bg": "#fff1f3",
    "accent_tint_border": "#fbcfd8",
    "font_ui": "'Inter', 'Hanken Grotesk', system-ui, -apple-system, sans-serif",
    "font_mono": "'JetBrains Mono', ui-monospace, 'SFMono-Regular', monospace",
}

# Official source documents surfaced as chips / citation cards in the UI.
LEGAL_SOURCE_DOCUMENTS: tuple[str, ...] = (
    "Federal Decree-Law No. 33 of 2021",
    "WPS Regulations",
    "MOHRE Directives",
)

LOCAL_MODEL_CHIP: str = "Qwen2.5 INT8 Offline"
OFFLINE_BADGE_TEXT: str = "100% Air-Gapped"

# =========================================================================== #
# UI copy (Phase 6). English is the source of truth; missing translations fall
# back to English via `ui_text()`. Chat *answers* are localised by the pipeline.
# =========================================================================== #
UI_STRINGS: dict[str, dict[str, str]] = {
    "app_tagline": {
        "en": "Offline UAE Labour Rights Assistant",
        "hi": "ऑफ़लाइन यूएई श्रम अधिकार सहायक",
        "ur": "آف لائن یو اے ای محنت حقوق معاون",
        "ml": "ഓഫ്‌ലൈൻ യുഎഇ തൊഴിൽ അവകാശ സഹായി",
        "ar": "مساعد حقوق العمل في الإمارات دون اتصال",
    },
    "choose_language": {
        "en": "Choose your language", "hi": "अपनी भाषा चुनें",
        "ur": "اپنی زبان منتخب کریں", "ml": "നിങ്ങളുടെ ഭാഷ തിരഞ്ഞെടുക്കുക",
        "ar": "اختر لغتك",
    },
    "enter_pin": {
        "en": "Enter your 4-digit PIN", "hi": "अपना 4-अंकों का पिन डालें",
        "ur": "اپنا 4 ہندسوں کا پن درج کریں", "ml": "നിങ്ങളുടെ 4-അക്ക പിൻ നൽകുക",
        "ar": "أدخل رقم التعريف الشخصي المكوّن من 4 أرقام",
    },
    "create_pin": {
        "en": "Create a 4-digit PIN", "hi": "4-अंकों का पिन बनाएँ",
        "ur": "4 ہندسوں کا پن بنائیں", "ml": "ഒരു 4-അക്ക പിൻ ഉണ്ടാക്കുക",
        "ar": "أنشئ رقم تعريف شخصي من 4 أرقام",
    },
    "confirm_pin": {
        "en": "Confirm PIN", "hi": "पिन की पुष्टि करें", "ur": "پن کی تصدیق کریں",
        "ml": "പിൻ സ്ഥിരീകരിക്കുക", "ar": "تأكيد رقم التعريف الشخصي",
    },
    "sign_in": {
        "en": "Sign in", "hi": "साइन इन करें", "ur": "سائن ان کریں",
        "ml": "സൈൻ ഇൻ ചെയ്യുക", "ar": "تسجيل الدخول",
    },
    "create_account": {
        "en": "Set up PIN", "hi": "पिन सेट करें", "ur": "پن ترتیب دیں",
        "ml": "പിൻ സജ്ജീകരിക്കുക", "ar": "إعداد رقم التعريف",
    },
    "continue_as_guest": {
        "en": "Continue as guest", "hi": "अतिथि के रूप में जारी रखें",
        "ur": "بطور مہمان جاری رکھیں", "ml": "അതിഥിയായി തുടരുക",
        "ar": "المتابعة كضيف",
    },
    "chat_title": {
        "en": "Ask about your rights", "hi": "अपने अधिकारों के बारे में पूछें",
        "ur": "اپنے حقوق کے بارے میں پوچھیں", "ml": "നിങ്ങളുടെ അവകാശങ്ങളെക്കുറിച്ച് ചോദിക്കുക",
        "ar": "اسأل عن حقوقك",
    },
    "ask_placeholder": {
        "en": "Ask a follow-up about UAE labour law…",
        "hi": "यूएई श्रम कानून के बारे में पूछें…",
        "ur": "یو اے ای محنت قانون کے بارے میں پوچھیں…",
        "ml": "യുഎഇ തൊഴിൽ നിയമത്തെക്കുറിച്ച് ചോദിക്കുക…",
        "ar": "اطرح سؤالاً عن قانون العمل الإماراتي…",
    },
    "answer_eyebrow": {
        "en": "ANSWER", "hi": "उत्तर", "ur": "جواب", "ml": "ഉത്തരം", "ar": "الإجابة",
    },
    "sources_panel_title": {
        "en": "Legal Sources & Citations (MOHRE / Decree-Law 33)",
        "hi": "कानूनी स्रोत और उद्धरण (MOHRE / डिक्री-लॉ 33)",
        "ur": "قانونی ذرائع و حوالہ جات (MOHRE / ڈکری لا 33)",
        "ml": "നിയമ സ്രോതസ്സുകളും ഉദ്ധരണികളും (MOHRE / ഡിക്രി-ലോ 33)",
        "ar": "المصادر القانونية والاستشهادات (وزارة الموارد البشرية / المرسوم 33)",
    },
    "show_all_sources": {
        "en": "Show all verified legal sources",
        "hi": "सभी सत्यापित कानूनी स्रोत दिखाएँ",
        "ur": "تمام تصدیق شدہ قانونی ذرائع دکھائیں",
        "ml": "പരിശോധിച്ച എല്ലാ നിയമ സ്രോതസ്സുകളും കാണിക്കുക",
        "ar": "عرض جميع المصادر القانونية المُوثّقة",
    },
    "helpline": {
        "en": "Call MOHRE 80084", "hi": "MOHRE 80084 पर कॉल करें",
        "ur": "MOHRE 80084 پر کال کریں", "ml": "MOHRE 80084-ൽ വിളിക്കുക",
        "ar": "اتصل بـ MOHRE على 80084",
    },
    "nav_chat": {
        "en": "Chat", "hi": "चैट", "ur": "چیٹ", "ml": "ചാറ്റ്", "ar": "المحادثة",
    },
    "nav_settings": {
        "en": "Settings", "hi": "सेटिंग्स", "ur": "ترتیبات", "ml": "ക്രമീകരണങ്ങൾ",
        "ar": "الإعدادات",
    },
    "sign_out": {
        "en": "Sign out", "hi": "साइन आउट", "ur": "سائن آؤٹ", "ml": "സൈൻ ഔട്ട്",
        "ar": "تسجيل الخروج",
    },
    "settings_title": {
        "en": "Profile & Settings", "hi": "प्रोफ़ाइल और सेटिंग्स",
        "ur": "پروفائل اور ترتیبات", "ml": "പ്രൊഫൈലും ക്രമീകരണങ്ങളും",
        "ar": "الملف الشخصي والإعدادات",
    },
    "language_label": {
        "en": "Language", "hi": "भाषा", "ur": "زبان", "ml": "ഭാഷ", "ar": "اللغة",
    },
    "font_size_label": {
        "en": "Text size", "hi": "पाठ का आकार", "ur": "متن کا سائز",
        "ml": "വാചക വലുപ്പം", "ar": "حجم النص",
    },
    "contract_type_label": {
        "en": "Contract type", "hi": "अनुबंध प्रकार", "ur": "معاہدے کی قسم",
        "ml": "കരാർ തരം", "ar": "نوع العقد",
    },
    "wipe_button": {
        "en": "Clear all data on this device",
        "hi": "इस डिवाइस पर सभी डेटा हटाएँ",
        "ur": "اس ڈیوائس پر تمام ڈیٹا صاف کریں",
        "ml": "ഈ ഉപകരണത്തിലെ എല്ലാ ഡാറ്റയും മായ്ക്കുക",
        "ar": "مسح جميع البيانات على هذا الجهاز",
    },
    "wipe_done": {
        "en": "All local data was erased.", "hi": "सभी स्थानीय डेटा मिटा दिया गया।",
        "ur": "تمام مقامی ڈیٹا مٹا دیا گیا۔", "ml": "എല്ലാ പ്രാദേശിക ഡാറ്റയും മായ്ച്ചു.",
        "ar": "تم مسح جميع البيانات المحلية.",
    },
    "kb_not_ready": {
        "en": (
            "The offline legal knowledge base has not been built on this device "
            "yet. Please contact MOHRE at 80084 or visit www.mohre.gov.ae."
        ),
        "hi": (
            "इस डिवाइस पर ऑफ़लाइन कानूनी ज्ञान-आधार अभी तैयार नहीं हुआ है। कृपया "
            "MOHRE से 80084 पर संपर्क करें या www.mohre.gov.ae देखें।"
        ),
        "ur": (
            "اس ڈیوائس پر آف لائن قانونی علمی بنیاد ابھی تیار نہیں ہوئی۔ براہِ کرم "
            "MOHRE سے 80084 پر رابطہ کریں یا www.mohre.gov.ae دیکھیں۔"
        ),
        "ml": (
            "ഈ ഉപകരണത്തിൽ ഓഫ്‌ലൈൻ നിയമ വിജ്ഞാന ശേഖരം ഇതുവരെ തയ്യാറാക്കിയിട്ടില്ല. "
            "ദയവായി MOHRE-യെ 80084-ൽ ബന്ധപ്പെടുക അല്ലെങ്കിൽ www.mohre.gov.ae സന്ദർശിക്കുക."
        ),
        "ar": (
            "لم يتم بعد إنشاء قاعدة المعرفة القانونية دون اتصال على هذا الجهاز. "
            "يُرجى الاتصال بـ MOHRE على 80084 أو زيارة www.mohre.gov.ae."
        ),
    },
}


# =========================================================================== #
# Helpers
# =========================================================================== #
def ui_text(key: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Localised UI string for ``key``; falls back to English then to the key."""
    entry = UI_STRINGS.get(key, {})
    return entry.get(language) or entry.get(DEFAULT_LANGUAGE) or key



def ensure_directories() -> None:
    """Create every runtime directory in :data:`ALL_DIRECTORIES` if missing.

    Idempotent and safe to call on every startup. Does not touch ``RAW_DIR``
    contents or any user data.
    """
    for directory in ALL_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def is_rtl(language_code: str) -> bool:
    """Return True if the given language code renders right-to-left."""
    return language_code in RTL_LANGUAGES


def language_name(language_code: str) -> str:
    """Native display name for a language code, falling back to the default."""
    return SUPPORTED_LANGUAGES.get(
        language_code, SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE]
    )
