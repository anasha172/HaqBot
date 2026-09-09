"""Phase 6 — Streamlit view interactions (app.py) via AppTest.

Runs the real app in-process (no server, no network) and drives the three
screens: login, chat with rendered sources/citations, and settings incl.
font size, language, contract type and the one-tap data wipe.
"""

from __future__ import annotations

import pytest

from src import config
from src.pipeline import Answer


class UiPipeline:
    """Returns a grounded answer with real sources so _render_turn is exercised."""

    def answer(self, query, *, language="en", k=None):
        class _Src:
            text = ("A full-time foreign worker is entitled to an end of service "
                    "gratuity after one year of continuous service.")
            score = 0.88
            citation = "[Article 51, Clause 1]"
            metadata = {"source_title": "Federal Decree-Law No. 33 of 2021",
                        "article_number": "51"}

        return Answer(
            text="You are entitled to a gratuity. [Article 51, Clause 1] "
                 "It also references [Article 77, Clause 2].",
            language=language,
            citations=["[Article 51, Clause 1]", "[Article 77, Clause 2]"],
            sources=[_Src()],
            used_fallback=False,
            top_score=0.88,
            elapsed_ms=2.0,
            disclaimer=config.LOCALIZED_DISCLAIMER[language],
            unsupported_citations=["[Article 77, Clause 2]"],
        )


@pytest.fixture
def at(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    idx = tmp_path / "idx"
    idx.mkdir()
    (idx / config.FAISS_INDEX_PATH.name).write_bytes(b"stub")
    (idx / config.VECTORSTORE_DOCS_PATH.name).write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("HAQBOT_DB_PATH", str(tmp_path / "local_user.db"))
    monkeypatch.setenv("HAQBOT_INDEX_DIR", str(idx))
    monkeypatch.setattr(
        "src.pipeline.build_default_pipeline", lambda **kw: UiPipeline()
    )

    app = AppTest.from_file(str(config.BASE_DIR / "app.py"), default_timeout=60)
    app.run()
    return app


def _click(app, label):
    [b for b in app.button if b.label == label][0].click().run()


def _register(app, pin="8351"):
    app.text_input(key="reg_user").set_value("demo")
    app.text_input(key="reg_pin").set_value(pin)
    app.text_input(key="reg_confirm").set_value(pin)
    _click(app, config.ui_text("create_account", "en"))


# --------------------------------------------------------------------------- #
class TestAuthScreen:
    def test_pin_mismatch_shows_error(self, at):
        at.text_input(key="reg_user").set_value("demo")
        at.text_input(key="reg_pin").set_value("8351")
        at.text_input(key="reg_confirm").set_value("9999")
        _click(at, config.ui_text("create_account", "en"))
        assert at.error
        assert at.session_state["screen"] == "auth"

    def test_weak_pin_shows_error(self, at):
        at.text_input(key="reg_user").set_value("demo")
        at.text_input(key="reg_pin").set_value("1234")
        at.text_input(key="reg_confirm").set_value("1234")
        _click(at, config.ui_text("create_account", "en"))
        assert at.error

    def test_register_then_login_after_signout(self, at):
        _register(at)
        assert at.session_state["screen"] == "chat"
        at.segmented_control[0].set_value(config.ui_text("sign_out", "en")).run()
        assert at.session_state["screen"] == "auth"
        # now the login form is shown
        at.text_input(key="login_user").set_value("demo")
        at.text_input(key="login_pin").set_value("8351")
        _click(at, config.ui_text("sign_in", "en"))
        assert at.session_state["screen"] == "chat"

    def test_bad_login_shows_error(self, at):
        _register(at)
        at.segmented_control[0].set_value(config.ui_text("sign_out", "en")).run()
        at.text_input(key="login_user").set_value("demo")
        at.text_input(key="login_pin").set_value("0000")
        _click(at, config.ui_text("sign_in", "en"))
        assert at.error


# --------------------------------------------------------------------------- #
class TestChatScreen:
    def test_answer_renders_with_citations_sources_and_warning(self, at):
        _register(at)
        at.chat_input[0].set_value("end of service gratuity?").run()
        assert not at.exception
        turn = at.session_state["chat_history"][0]
        assert turn["answer"].used_fallback is False
        blob = " ".join(m.value for m in at.markdown)
        assert "haq-query-heading" in blob
        assert "[Article 51, Clause 1]" in blob
        assert "haq-source-card" in blob
        # [Article 77 ...] is not in the retrieved set -> unsupported warning
        assert at.warning

    def test_empty_prompt_is_ignored(self, at):
        _register(at)
        at.chat_input[0].set_value("   ").run()
        assert at.session_state["chat_history"] == []


# --------------------------------------------------------------------------- #
class TestSettingsScreen:
    def _go_settings(self, at):
        _register(at)
        at.segmented_control[0].set_value(
            config.ui_text("nav_settings", "en")
        ).run()

    def test_change_font_size_and_language(self, at):
        self._go_settings(at)
        at.select_slider(key="settings_font").set_value("Large").run()
        assert at.session_state["font_scale"] == "Large"
        at.selectbox(key="settings_lang").set_value("ur").run()
        assert at.session_state["language"] == "ur"
        assert at.session_state["_haq_svc"].auth.get_public_profile(
            "demo"
        ).language == "ur"

    def test_set_contract_type(self, at):
        self._go_settings(at)
        at.selectbox(key="settings_contract").set_value("Limited").run()
        assert not at.exception
        assert at.session_state["session"].profile.contract_type == "Limited"

    def test_wipe_requires_confirm_then_clears(self, at):
        self._go_settings(at)
        wipe_label = config.ui_text("wipe_button", "en")
        assert [b for b in at.button if b.label == wipe_label][0].disabled is True
        at.checkbox[0].set_value(True).run()
        [b for b in at.button if b.label == wipe_label][0].click().run()
        assert not at.exception
        assert at.session_state["_haq_svc"].auth.list_users() == []
        assert at.session_state["screen"] == "auth"

    def test_back_to_chat(self, at):
        self._go_settings(at)
        _click(at, f"← {config.ui_text('nav_chat', 'en')}")
        assert at.session_state["screen"] == "chat"


# --------------------------------------------------------------------------- #
class TestGuestAndChrome:
    def test_guest_cannot_set_contract_type(self, at):
        _click(at, config.ui_text("continue_as_guest", "en"))
        at.segmented_control[0].set_value(
            config.ui_text("nav_settings", "en")
        ).run()
        keys = {sb.key for sb in at.selectbox}
        assert "settings_contract" not in keys  # hidden for guests
        captions = " ".join(c.value for c in at.caption)
        assert "contract type securely" in captions

    def test_helpline_fab_and_offline_badge_on_every_screen(self, at):
        _register(at)
        blob = " ".join(m.value for m in at.markdown)
        assert f'href="tel:{config.MOHRE_HELPLINE}"' in blob
        assert config.OFFLINE_BADGE_TEXT in blob
