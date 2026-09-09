"""HaqBot — mobile-first, offline multilingual legal RAG assistant.

Thin Streamlit view over :class:`src.appservice.AppService`. Three screens:
Local Auth / PIN  ->  Multilingual Chat  ->  Profile & Settings.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os

from src import config

# Air-gap: block huggingface_hub / transformers network access before anything
# heavy is imported.
for _k, _v in config.OFFLINE_ENV.items():
    os.environ.setdefault(_k, _v)

import streamlit as st  # noqa: E402

from src.appservice import (  # noqa: E402
    AppError,
    AppService,
    SCREEN_AUTH,
    SCREEN_CHAT,
    SCREEN_SETTINGS,
    build_css,
)

st.set_page_config(
    page_title="HaqBot — حق بوت",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def make_service() -> AppService:
    # Optional relocation of on-device data (used by tests; harmless in prod).
    db_path = os.environ.get("HAQBOT_DB_PATH") or None
    index_dir = os.environ.get("HAQBOT_INDEX_DIR") or None

    def _pipeline_factory():
        from src.pipeline import build_default_pipeline

        return build_default_pipeline(index_dir=index_dir)

    return AppService(
        pipeline_factory=_pipeline_factory, db_path=db_path, index_dir=index_dir
    )


state = st.session_state
if "_haq_svc" not in state:
    state["_haq_svc"] = make_service()
svc: AppService = state["_haq_svc"]
svc.init_session(state)


def t(key: str) -> str:
    return config.ui_text(key, state["language"])


def _lang_selectbox(container, *, key: str) -> None:
    codes = list(config.SUPPORTED_LANGUAGES)
    current = state["language"]
    choice = container.selectbox(
        t("language_label"),
        codes,
        index=codes.index(current) if current in codes else 0,
        format_func=lambda c: config.SUPPORTED_LANGUAGES[c],
        key=key,
    )
    if choice != current:
        try:
            svc.set_language(state, choice)
        except AppError as exc:
            container.error(str(exc))
        st.rerun()


# --------------------------------------------------------------------------- #
# Chrome
# --------------------------------------------------------------------------- #
st.markdown(
    f"<style>{build_css(font_scale=svc.font_scale_value(state), rtl=svc.is_rtl(state))}</style>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="haq-header"><span class="haq-brand-glyph">⚖️</span>'
    f'<span>{config.APP_NAME}</span>'
    f'<span class="haq-offline-badge">● {config.OFFLINE_BADGE_TEXT}</span></div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Screen: Auth / PIN
# --------------------------------------------------------------------------- #
def render_auth() -> None:
    st.markdown(f'<div class="haq-eyebrow">{config.APP_NAME_NATIVE}</div>', unsafe_allow_html=True)
    st.markdown(f"### {t('app_tagline')}")
    _lang_selectbox(st, key="auth_lang")

    if state.get("auth_error"):
        st.error(state["auth_error"])
        state["auth_error"] = ""

    has_account = svc.has_account()
    st.write("")

    if has_account:
        with st.form("login_form"):
            username = st.text_input("Name", key="login_user")
            pin = st.text_input(
                t("enter_pin"), type="password", max_chars=config.PIN_LENGTH,
                key="login_pin",
            )
            submitted = st.form_submit_button(t("sign_in"), type="primary")
        if submitted:
            try:
                svc.login(state, username, pin)
                st.rerun()
            except AppError as exc:
                st.error(str(exc))
    else:
        st.caption("First time on this device — create a local PIN. It never leaves your phone.")
        with st.form("register_form"):
            username = st.text_input("Name", key="reg_user")
            pin = st.text_input(
                t("create_pin"), type="password", max_chars=config.PIN_LENGTH,
                key="reg_pin",
            )
            confirm = st.text_input(
                t("confirm_pin"), type="password", max_chars=config.PIN_LENGTH,
                key="reg_confirm",
            )
            submitted = st.form_submit_button(t("create_account"), type="primary")
        if submitted:
            try:
                svc.register(state, username, pin, confirm)
                st.rerun()
            except AppError as exc:
                st.error(str(exc))

    st.divider()
    if st.button(t("continue_as_guest")):
        svc.login_guest(state)
        st.rerun()


# --------------------------------------------------------------------------- #
# Screen: Chat
# --------------------------------------------------------------------------- #
def _render_sources(turn: dict) -> None:
    answer = turn["answer"]
    with st.expander(t("sources_panel_title"), expanded=bool(answer.sources)):
        if not answer.sources:
            st.caption(
                "No verified legal sources matched — see the MOHRE referral above."
            )
        for i, src in enumerate(answer.sources, start=1):
            active = "is-active" if i == 1 and not answer.used_fallback else ""
            doc = src.metadata.get("source_title", "") or src.metadata.get("doc_title", "")
            snippet = (src.text[:180] + "…") if len(src.text) > 180 else src.text
            st.markdown(
                f'<div class="haq-source-card {active}">'
                f'<span class="haq-source-badge">{i}</span>'
                f'<span class="haq-source-doc">{doc}</span><br>'
                f'<strong>{src.citation}</strong><br>{snippet}</div>',
                unsafe_allow_html=True,
            )
        st.caption(t("show_all_sources"))


def _render_turn(turn: dict) -> None:
    answer = turn["answer"]
    st.markdown(f'<div class="haq-eyebrow">{t("answer_eyebrow")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="haq-query-heading">{turn["query"]}</div>', unsafe_allow_html=True)
    st.markdown(answer.text)

    if answer.citations:
        chips = " ".join(
            f'<span class="haq-chip"><span class="haq-cite">{n}</span>{c}</span>'
            for n, c in enumerate(answer.citations, start=1)
        )
        st.markdown(chips, unsafe_allow_html=True)
    if answer.unsupported_citations:
        st.warning(
            "Some citations could not be matched to the offline text: "
            + ", ".join(answer.unsupported_citations)
        )
    _render_sources(turn)
    st.markdown(f'<div class="haq-disclaimer">{answer.disclaimer}</div>', unsafe_allow_html=True)
    st.divider()


def render_chat() -> None:
    nav = st.segmented_control(
        "nav", [t("nav_chat"), t("nav_settings"), t("sign_out")],
        default=t("nav_chat"), label_visibility="collapsed",
    )
    if nav == t("nav_settings"):
        svc.navigate(state, SCREEN_SETTINGS)
        st.rerun()
    elif nav == t("sign_out"):
        svc.logout(state)
        st.rerun()

    if not svc.kb_ready:
        st.info(config.ui_text("kb_not_ready", state["language"]))

    if not state["chat_history"]:
        st.markdown(f"## {t('chat_title')}")
        st.caption(
            "Ask about wages, end-of-service gratuity, leave, probation, "
            "contract types, or the Wage Protection System."
        )

    for turn in state["chat_history"]:
        _render_turn(turn)

    st.markdown(
        f'<span class="haq-model-chip">▮ {config.LOCAL_MODEL_CHIP}</span>',
        unsafe_allow_html=True,
    )
    prompt = st.chat_input(t("ask_placeholder"))
    if prompt:
        try:
            svc.ask(state, prompt)
        except AppError as exc:
            st.error(str(exc))
        st.rerun()


# --------------------------------------------------------------------------- #
# Screen: Profile & Settings
# --------------------------------------------------------------------------- #
def render_settings() -> None:
    if st.button(f"← {t('nav_chat')}"):
        svc.navigate(state, SCREEN_CHAT)
        st.rerun()

    st.markdown(f"## {t('settings_title')}")
    _lang_selectbox(st, key="settings_lang")

    labels = list(config.FONT_SCALE_OPTIONS)
    current_scale = state["font_scale"]
    new_scale = st.select_slider(
        t("font_size_label"), options=labels,
        value=current_scale if current_scale in labels else config.DEFAULT_FONT_SCALE,
        key="settings_font",
    )
    if new_scale != current_scale:
        svc.set_font_scale(state, new_scale)
        st.rerun()

    session = state.get("session")
    if session is not None and not session.is_guest:
        types = list(config.CONTRACT_TYPES)
        current_ct = session.profile.contract_type or "Not specified"
        new_ct = st.selectbox(
            t("contract_type_label"), types,
            index=types.index(current_ct) if current_ct in types else 0,
            key="settings_contract",
        )
        if new_ct != current_ct:
            try:
                svc.set_contract_type(state, new_ct)
            except AppError as exc:
                st.error(str(exc))
            st.rerun()
    else:
        st.caption("Sign in with a PIN to save your contract type securely.")

    info = svc.app_info()
    st.divider()
    st.caption(
        f"{info.app_name} v{info.version}  ·  offline knowledge base snapshot: "
        f"{info.kb_snapshot}  ·  index built: {'yes' if info.kb_ready else 'no'}"
    )

    st.divider()
    st.subheader("Privacy")
    st.write(
        "Everything — your PIN, settings, and chats — is stored only on this "
        "device. Nothing is ever sent anywhere."
    )
    confirm = st.checkbox("Yes, erase everything on this device")
    if st.button(t("wipe_button"), type="primary", disabled=not confirm):
        svc.wipe_all_data(state)
        st.success(t("wipe_done"))
        st.rerun()


# --------------------------------------------------------------------------- #
# Router + helpline FAB
# --------------------------------------------------------------------------- #
_SCREEN = svc.current_screen(state)
if _SCREEN == SCREEN_AUTH:
    render_auth()
elif _SCREEN == SCREEN_CHAT:
    render_chat()
elif _SCREEN == SCREEN_SETTINGS:
    render_settings()

st.markdown(
    f'<a class="haq-fab" href="tel:{config.MOHRE_HELPLINE}">📞 {t("helpline")}</a>',
    unsafe_allow_html=True,
)
