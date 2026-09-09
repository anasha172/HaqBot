"""HaqBot — Streamlit-free application controller.

All screen logic, auth wiring, RAG calls, settings persistence, the one-tap data
purge, and CSS generation live here so they can be unit-tested without a running
Streamlit server. ``app.py`` is a thin view layer over this class.

Session state is any ``MutableMapping`` (``st.session_state`` in production, a
plain ``dict`` in tests).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from src import config
from src.auth import AuthError, AuthManager, AuthSession
from src.pipeline import Answer, HaqBotPipeline, PipelineError

# Enforce the air-gap the instant this module is imported, before any lazy
# heavy import downstream (transformers / huggingface_hub) can run.
for _key, _value in config.OFFLINE_ENV.items():
    os.environ.setdefault(_key, _value)

__all__ = ["AppError", "AppService", "build_css", "SCREEN_AUTH", "SCREEN_CHAT",
           "SCREEN_SETTINGS"]

SCREEN_AUTH = "auth"
SCREEN_CHAT = "chat"
SCREEN_SETTINGS = "settings"
_SCREENS = (SCREEN_AUTH, SCREEN_CHAT, SCREEN_SETTINGS)

PipelineFactory = Callable[[], HaqBotPipeline]


class AppError(Exception):
    """A user-safe application-level error (bad input, blocked action)."""


# --------------------------------------------------------------------------- #
# CSS — Perplexity-style rose light theme, mobile-first
# --------------------------------------------------------------------------- #
def build_css(*, font_scale: float = 1.0, rtl: bool = False) -> str:
    """Return the app's stylesheet. No external fonts/URLs are referenced."""
    t = config.THEME
    scale = max(0.8, min(1.6, float(font_scale)))
    direction = "rtl" if rtl else "ltr"
    sys_ui = (
        "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, "
        "'Helvetica Neue', Arial, 'Noto Sans', sans-serif"
    )
    sys_mono = "ui-monospace, 'SFMono-Regular', 'Cascadia Mono', Menlo, monospace"
    return f"""
:root {{
  --haq-canvas:{t['canvas']}; --haq-card:{t['card']}; --haq-ink:{t['ink']};
  --haq-meta:{t['meta']}; --haq-border:{t['border']}; --haq-fill:{t['fill']};
  --haq-accent:{t['accent']}; --haq-accent-hover:{t['accent_hover']};
  --haq-tint-bg:{t['accent_tint_bg']}; --haq-tint-border:{t['accent_tint_border']};
  --haq-scale:{scale};
}}
html, body, [data-testid="stAppViewContainer"] {{
  background:var(--haq-canvas); color:var(--haq-ink);
  font-family:{sys_ui}; direction:{direction};
  font-size:calc(1rem * var(--haq-scale));
}}
#MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"], footer {{
  visibility:hidden; height:0;
}}
.block-container {{ max-width:760px; padding-top:4.2rem; padding-bottom:6rem; }}
.haq-header {{
  position:fixed; top:0; left:0; right:0; z-index:999; background:var(--haq-card);
  border-bottom:1px solid var(--haq-border); padding:.6rem 1rem;
  display:flex; align-items:center; gap:.6rem; font-weight:700;
}}
.haq-brand-glyph {{ color:var(--haq-accent); font-size:1.15rem; }}
.haq-offline-badge {{
  margin-inline-start:auto; font-size:.72rem; font-weight:600; color:var(--haq-meta);
  border:1px solid var(--haq-border); border-radius:999px; padding:.15rem .55rem;
  background:var(--haq-fill);
}}
.haq-eyebrow {{
  color:var(--haq-accent); font-weight:700; font-size:.72rem;
  letter-spacing:.14em; margin-bottom:.2rem;
}}
.haq-query-heading {{ font-size:1.5rem; font-weight:800; line-height:1.25; margin:.1rem 0 1rem; }}
.haq-cite {{
  font-family:{sys_mono}; color:var(--haq-accent); font-weight:600;
  font-size:.8em; vertical-align:super;
}}
.haq-chip {{
  display:inline-flex; align-items:center; gap:.35rem; background:var(--haq-card);
  border:1px solid var(--haq-border); border-radius:999px; padding:.25rem .7rem;
  margin:.2rem .3rem .2rem 0; font-size:.8rem; color:var(--haq-ink);
}}
.haq-chip .haq-cite {{ vertical-align:baseline; }}
.haq-source-card {{
  background:var(--haq-card); border:1px solid var(--haq-border);
  border-radius:12px; padding:.7rem .8rem; margin-bottom:.55rem;
}}
.haq-source-card.is-active {{ background:var(--haq-tint-bg); border-color:var(--haq-tint-border); }}
.haq-source-doc {{ font-family:{sys_mono}; font-size:.78rem; color:var(--haq-meta); }}
.haq-source-badge {{
  display:inline-block; min-width:1.4rem; text-align:center; border-radius:999px;
  background:var(--haq-accent); color:#fff; font-family:{sys_mono};
  font-size:.72rem; padding:.05rem .35rem; margin-inline-end:.4rem;
}}
.haq-disclaimer {{ color:var(--haq-meta); font-size:.8rem; border-top:1px solid var(--haq-border); padding-top:.6rem; margin-top:1rem; }}
.haq-model-chip {{
  font-family:{sys_mono}; font-size:.72rem; color:var(--haq-meta);
  border:1px solid var(--haq-border); border-radius:999px; padding:.1rem .5rem;
  background:var(--haq-fill);
}}
.haq-fab {{
  position:fixed; inset-block-end:1rem; inset-inline-end:1rem; z-index:1000;
  background:var(--haq-accent); color:#fff !important; text-decoration:none;
  border-radius:999px; padding:.7rem 1.05rem; font-weight:700; font-size:.9rem;
  box-shadow:0 6px 18px rgba(225,29,72,.35);
}}
.stButton > button {{ border-radius:10px; font-weight:600; }}
.stButton > button[kind="primary"] {{ background:var(--haq-accent); border-color:var(--haq-accent); }}
@media (max-width:640px) {{
  .haq-query-heading {{ font-size:1.25rem; }}
  .block-container {{ padding-left:.8rem; padding-right:.8rem; }}
}}
""".strip()


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
@dataclass
class AppInfo:
    app_name: str
    version: str
    kb_snapshot: str
    kb_ready: bool
    offline: bool
    model_chip: str


class AppService:
    def __init__(
        self,
        *,
        auth_manager: AuthManager | None = None,
        pipeline_factory: PipelineFactory | None = None,
        db_path: str | Path | None = None,
        index_dir: str | Path | None = None,
    ) -> None:
        self.auth = auth_manager or AuthManager(db_path or config.DB_PATH)
        self._pipeline_factory = pipeline_factory
        self._pipeline: HaqBotPipeline | None = None
        self.index_dir = Path(index_dir or config.VECTORSTORE_DIR)

    # -- knowledge base state ---------------------------------------------------
    @property
    def kb_ready(self) -> bool:
        return (
            (self.index_dir / config.FAISS_INDEX_PATH.name).is_file()
            and (self.index_dir / config.VECTORSTORE_DOCS_PATH.name).is_file()
        )

    def app_info(self) -> AppInfo:
        return AppInfo(
            app_name=config.APP_NAME,
            version=config.APP_VERSION,
            kb_snapshot=config.KB_SNAPSHOT_DATE,
            kb_ready=self.kb_ready,
            offline=config.OFFLINE_ONLY,
            model_chip=config.LOCAL_MODEL_CHIP,
        )

    # -- session bootstrap / routing -----------------------------------------
    def init_session(self, state: MutableMapping[str, Any]) -> None:
        if state.get("_haq_init"):
            return
        state.setdefault("screen", SCREEN_AUTH)
        state.setdefault("language", config.DEFAULT_LANGUAGE)
        state.setdefault("font_scale", config.DEFAULT_FONT_SCALE)
        state.setdefault("session", None)
        state.setdefault("chat_history", [])
        state.setdefault("auth_error", "")
        state["_haq_init"] = True

    def current_screen(self, state: MutableMapping[str, Any]) -> str:
        screen = state.get("screen", SCREEN_AUTH)
        if state.get("session") is None and screen != SCREEN_AUTH:
            return SCREEN_AUTH
        return screen if screen in _SCREENS else SCREEN_AUTH

    def navigate(self, state: MutableMapping[str, Any], screen: str) -> None:
        if screen not in _SCREENS:
            raise AppError("Unknown screen.")
        if screen != SCREEN_AUTH and state.get("session") is None:
            raise AppError("Please sign in first.")
        state["screen"] = screen

    # -- auth --------------------------------------------------------------
    def has_account(self) -> bool:
        return self.auth.has_any_pin_user()

    def _adopt_profile(self, state: MutableMapping[str, Any], session: AuthSession) -> None:
        state["session"] = session
        state["language"] = session.profile.language
        state["font_scale"] = session.profile.font_scale
        state["screen"] = SCREEN_CHAT
        state["auth_error"] = ""

    def register(
        self, state: MutableMapping[str, Any], username: str, pin: str, confirm: str
    ) -> None:
        if pin != confirm:
            raise AppError("The two PINs do not match.")
        try:
            self.auth.register(username, pin)
            session = self.auth.authenticate(username, pin)
        except AuthError as exc:
            raise AppError(str(exc)) from exc
        self._adopt_profile(state, session)

    def login(self, state: MutableMapping[str, Any], username: str, pin: str) -> None:
        try:
            session = self.auth.authenticate(username, pin)
        except AuthError as exc:
            raise AppError(str(exc)) from exc
        self._adopt_profile(state, session)

    def login_guest(self, state: MutableMapping[str, Any]) -> None:
        session = self.auth.login_guest()
        self._adopt_profile(state, session)
        state["language"] = state.get("language", config.DEFAULT_LANGUAGE)

    def logout(self, state: MutableMapping[str, Any]) -> None:
        state["session"] = None
        state["chat_history"] = []
        state["screen"] = SCREEN_AUTH

    def require_session(self, state: MutableMapping[str, Any]) -> AuthSession:
        session = state.get("session")
        if session is None:
            raise AppError("Please sign in first.")
        return session

    # -- chat -----------------------------------------------------------------
    def _get_pipeline(self) -> HaqBotPipeline | None:
        if self._pipeline is not None:
            return self._pipeline
        if self._pipeline_factory is None:
            return None
        self._pipeline = self._pipeline_factory()
        return self._pipeline

    def ask(self, state: MutableMapping[str, Any], query: str) -> Answer:
        self.require_session(state)
        language = state.get("language", config.DEFAULT_LANGUAGE)
        if query is None or not query.strip():
            raise AppError("Please type a question.")
        query = query.strip()

        pipeline = self._get_pipeline() if self.kb_ready else None
        if pipeline is None:
            answer = Answer(
                text=config.ui_text("kb_not_ready", language),
                language=language, citations=[], sources=[],
                used_fallback=True, top_score=-1.0, elapsed_ms=0.0,
                disclaimer=config.LOCALIZED_DISCLAIMER.get(
                    language, config.LEGAL_DISCLAIMER
                ),
            )
        else:
            try:
                answer = pipeline.answer(query, language=language)
            except PipelineError as exc:
                raise AppError(str(exc)) from exc

        state["chat_history"].append(
            {"query": query, "answer": answer, "language": language}
        )
        return answer

    def clear_chat(self, state: MutableMapping[str, Any]) -> None:
        state["chat_history"] = []

    # -- settings / profile --------------------------------------------------
    def set_language(self, state: MutableMapping[str, Any], code: str) -> None:
        if code not in config.SUPPORTED_LANGUAGES:
            raise AppError("Unsupported language.")
        state["language"] = code
        session = state.get("session")
        if session is not None:
            try:
                self.auth.update_profile(session, language=code)
            except AuthError as exc:  # pragma: no cover - defensive
                raise AppError(str(exc)) from exc

    def set_font_scale(self, state: MutableMapping[str, Any], label: str) -> None:
        if label not in config.FONT_SCALE_OPTIONS:
            raise AppError("Unsupported text size.")
        state["font_scale"] = label
        session = state.get("session")
        if session is not None:
            try:
                self.auth.update_profile(session, font_scale=label)
            except AuthError as exc:  # pragma: no cover - defensive
                raise AppError(str(exc)) from exc

    def set_contract_type(self, state: MutableMapping[str, Any], value: str) -> None:
        session = self.require_session(state)
        try:
            self.auth.update_profile(
                session, contract_type=None if value in ("", "Not specified") else value
            )
        except AuthError as exc:
            raise AppError(str(exc)) from exc

    def font_scale_value(self, state: MutableMapping[str, Any]) -> float:
        return config.FONT_SCALE_OPTIONS.get(
            state.get("font_scale", config.DEFAULT_FONT_SCALE), 1.0
        )

    def is_rtl(self, state: MutableMapping[str, Any]) -> bool:
        return config.is_rtl(state.get("language", config.DEFAULT_LANGUAGE))

    # -- one-tap privacy purge ---------------------------------------------
    def wipe_all_data(self, state: MutableMapping[str, Any]) -> None:
        """Erase the local database and every session value (keeps the service)."""
        self.auth.wipe_all()
        preserved = {k: state[k] for k in ("_haq_svc",) if k in state}
        for key in list(state.keys()):
            del state[key]
        state.update(preserved)
        self.init_session(state)
