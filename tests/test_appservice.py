"""Phase 6 — AppService controller (Streamlit-free) and CSS builder."""

from __future__ import annotations

import pytest

from src import config
from src.appservice import (
    SCREEN_AUTH,
    SCREEN_CHAT,
    SCREEN_SETTINGS,
    AppError,
    AppService,
    build_css,
)
from src.auth import AuthManager
from src.pipeline import Answer

GOOD_PIN = "8351"
OTHER_PIN = "2947"


class FakePipeline:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def answer(self, query, *, language="en", k=None):
        self.calls.append((query, language))
        return Answer(
            text=f"Grounded answer for '{query}'. [Article 51, Clause 1]",
            language=language,
            citations=["[Article 51, Clause 1]"],
            sources=[],
            used_fallback=False,
            top_score=0.91,
            elapsed_ms=1.0,
            disclaimer=config.LOCALIZED_DISCLAIMER[language],
        )


def _kb_ready_dir(tmp_path):
    d = tmp_path / "idx"
    d.mkdir()
    (d / config.FAISS_INDEX_PATH.name).write_bytes(b"stub-index")
    (d / config.VECTORSTORE_DOCS_PATH.name).write_text("{}\n", encoding="utf-8")
    return d


@pytest.fixture
def svc(tmp_path):
    return AppService(
        auth_manager=AuthManager(tmp_path / "local_user.db"),
        pipeline_factory=None,
        index_dir=tmp_path / "idx-missing",
    )


@pytest.fixture
def state():
    return {}


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
class TestBuildCss:
    def test_contains_theme_tokens(self):
        css = build_css()
        assert config.THEME["accent"] in css          # #e11d48
        assert config.THEME["canvas"] in css          # #f5f6f8
        for cls in (".haq-header", ".haq-fab", ".haq-cite", ".haq-source-card.is-active"):
            assert cls in css

    def test_font_scale_is_applied_and_clamped(self):
        assert "--haq-scale:1.4" in build_css(font_scale=1.4)
        assert "--haq-scale:1.6" in build_css(font_scale=9.0)   # clamped high
        assert "--haq-scale:0.8" in build_css(font_scale=0.1)   # clamped low

    def test_direction_follows_rtl_flag(self):
        assert "direction:rtl" in build_css(rtl=True)
        assert "direction:ltr" in build_css(rtl=False)

    def test_no_external_urls(self):
        css = build_css()
        assert "http://" not in css and "https://" not in css
        assert "googleapis" not in css and "@import" not in css


# --------------------------------------------------------------------------- #
# Session bootstrap + routing
# --------------------------------------------------------------------------- #
class TestSessionRouting:
    def test_init_sets_defaults_and_is_idempotent(self, svc, state):
        svc.init_session(state)
        assert state["screen"] == SCREEN_AUTH
        assert state["language"] == config.DEFAULT_LANGUAGE
        assert state["chat_history"] == []
        state["screen"] = SCREEN_CHAT
        svc.init_session(state)  # must not reset
        assert state["screen"] == SCREEN_CHAT

    def test_current_screen_forces_auth_without_session(self, svc, state):
        svc.init_session(state)
        state["screen"] = SCREEN_CHAT
        assert svc.current_screen(state) == SCREEN_AUTH

    def test_navigate_requires_session(self, svc, state):
        svc.init_session(state)
        with pytest.raises(AppError):
            svc.navigate(state, SCREEN_SETTINGS)

    def test_navigate_rejects_unknown_screen(self, svc, state):
        svc.init_session(state)
        state["session"] = object()
        with pytest.raises(AppError):
            svc.navigate(state, "nowhere")


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
class TestAuth:
    def test_register_then_state_is_authed_on_chat(self, svc, state):
        svc.init_session(state)
        assert svc.has_account() is False
        svc.register(state, "ravi", GOOD_PIN, GOOD_PIN)
        assert svc.has_account() is True
        assert state["session"] is not None
        assert state["screen"] == SCREEN_CHAT

    def test_register_pin_mismatch(self, svc, state):
        svc.init_session(state)
        with pytest.raises(AppError):
            svc.register(state, "ravi", GOOD_PIN, "0000")

    def test_register_weak_pin_surfaces_auth_message(self, svc, state):
        svc.init_session(state)
        with pytest.raises(AppError):
            svc.register(state, "ravi", "1234", "1234")

    def test_login_wrong_pin(self, svc, state):
        svc.init_session(state)
        svc.register(state, "ravi", GOOD_PIN, GOOD_PIN)
        svc.logout(state)
        with pytest.raises(AppError):
            svc.login(state, "ravi", OTHER_PIN)

    def test_login_adopts_saved_language(self, svc, state):
        svc.init_session(state)
        svc.register(state, "ravi", GOOD_PIN, GOOD_PIN)
        svc.set_language(state, "ml")
        svc.logout(state)
        svc.login(state, "ravi", GOOD_PIN)
        assert state["language"] == "ml"

    def test_guest_login(self, svc, state):
        svc.init_session(state)
        svc.login_guest(state)
        assert state["session"].is_guest is True
        assert state["screen"] == SCREEN_CHAT

    def test_logout_clears_everything(self, svc, state):
        svc.init_session(state)
        svc.register(state, "ravi", GOOD_PIN, GOOD_PIN)
        state["chat_history"].append({"x": 1})
        svc.logout(state)
        assert state["session"] is None
        assert state["chat_history"] == []
        assert state["screen"] == SCREEN_AUTH

    def test_lockout_message_propagates(self, svc, state):
        svc.init_session(state)
        svc.register(state, "ravi", GOOD_PIN, GOOD_PIN)
        svc.logout(state)
        for _ in range(config.MAX_PIN_ATTEMPTS):
            with pytest.raises(AppError):
                svc.login(state, "ravi", OTHER_PIN)
        with pytest.raises(AppError) as exc:
            svc.login(state, "ravi", GOOD_PIN)
        assert "attempts" in str(exc.value).lower() or "minute" in str(exc.value).lower()


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
class TestChat:
    def test_ask_requires_session(self, svc, state):
        svc.init_session(state)
        with pytest.raises(AppError):
            svc.ask(state, "hello")

    def test_ask_rejects_empty(self, svc, state):
        svc.init_session(state)
        svc.login_guest(state)
        with pytest.raises(AppError):
            svc.ask(state, "   ")

    def test_ask_without_kb_returns_fallback_and_skips_pipeline(self, tmp_path, state):
        factory_called = {"n": 0}

        def _factory():
            factory_called["n"] += 1
            return FakePipeline()

        svc = AppService(
            auth_manager=AuthManager(tmp_path / "u.db"),
            pipeline_factory=_factory,
            index_dir=tmp_path / "no-index",
        )
        svc.init_session(state)
        svc.login_guest(state)
        answer = svc.ask(state, "what is gratuity")
        assert answer.used_fallback is True
        assert answer.text == config.ui_text("kb_not_ready", "en")
        assert factory_called["n"] == 0
        assert len(state["chat_history"]) == 1

    def test_ask_with_kb_uses_pipeline_and_passes_language(self, tmp_path, state):
        fake = FakePipeline()
        svc = AppService(
            auth_manager=AuthManager(tmp_path / "u.db"),
            pipeline_factory=lambda: fake,
            index_dir=_kb_ready_dir(tmp_path),
        )
        svc.init_session(state)
        svc.login_guest(state)
        svc.set_language(state, "hi")
        answer = svc.ask(state, "end of service gratuity")
        assert answer.used_fallback is False
        assert fake.calls == [("end of service gratuity", "hi")]
        assert state["chat_history"][0]["answer"] is answer

    def test_ask_pipeline_error_becomes_app_error(self, tmp_path, state):
        from src.pipeline import PipelineError

        class BoomPipeline:
            def answer(self, *a, **k):
                raise PipelineError("model down")

        svc = AppService(
            auth_manager=AuthManager(tmp_path / "u.db"),
            pipeline_factory=lambda: BoomPipeline(),
            index_dir=_kb_ready_dir(tmp_path),
        )
        svc.init_session(state)
        svc.login_guest(state)
        with pytest.raises(AppError):
            svc.ask(state, "hi")

    def test_clear_chat(self, svc, state):
        svc.init_session(state)
        svc.login_guest(state)
        svc.ask(state, "one")
        svc.clear_chat(state)
        assert state["chat_history"] == []

    def test_kb_ready_but_no_factory_falls_back(self, tmp_path, state):
        svc = AppService(
            auth_manager=AuthManager(tmp_path / "u.db"),
            pipeline_factory=None,
            index_dir=_kb_ready_dir(tmp_path),
        )
        svc.init_session(state)
        svc.login_guest(state)
        answer = svc.ask(state, "gratuity")
        assert answer.used_fallback is True

    def test_pipeline_is_built_once_and_reused(self, tmp_path, state):
        builds = {"n": 0}

        def _factory():
            builds["n"] += 1
            return FakePipeline()

        svc = AppService(
            auth_manager=AuthManager(tmp_path / "u.db"),
            pipeline_factory=_factory,
            index_dir=_kb_ready_dir(tmp_path),
        )
        svc.init_session(state)
        svc.login_guest(state)
        svc.ask(state, "one")
        svc.ask(state, "two")
        assert builds["n"] == 1


# --------------------------------------------------------------------------- #
# Settings / profile
# --------------------------------------------------------------------------- #
class TestSettings:
    def test_language_persists_to_profile(self, svc, state):
        svc.init_session(state)
        svc.register(state, "ravi", GOOD_PIN, GOOD_PIN)
        svc.set_language(state, "ur")
        assert svc.auth.get_public_profile("ravi").language == "ur"

    def test_invalid_language_and_font(self, svc, state):
        svc.init_session(state)
        svc.login_guest(state)
        with pytest.raises(AppError):
            svc.set_language(state, "fr")
        with pytest.raises(AppError):
            svc.set_font_scale(state, "Gigantic")

    def test_font_scale_persists_and_maps_to_value(self, svc, state):
        svc.init_session(state)
        svc.register(state, "ravi", GOOD_PIN, GOOD_PIN)
        svc.set_font_scale(state, "Large")
        assert state["font_scale"] == "Large"
        assert svc.font_scale_value(state) == config.FONT_SCALE_OPTIONS["Large"]
        assert svc.auth.get_public_profile("ravi").font_scale == "Large"

    def test_contract_type_guest_blocked_pin_user_ok(self, svc, state):
        svc.init_session(state)
        svc.login_guest(state)
        with pytest.raises(AppError):
            svc.set_contract_type(state, "Limited")
        svc.logout(state)
        svc.register(state, "ravi", GOOD_PIN, GOOD_PIN)
        svc.set_contract_type(state, "Limited")
        assert state["session"].profile.contract_type == "Limited"
        svc.set_contract_type(state, "Not specified")
        assert state["session"].profile.contract_type is None

    def test_is_rtl(self, svc, state):
        svc.init_session(state)
        svc.set_language(state, "ar")
        assert svc.is_rtl(state) is True
        svc.set_language(state, "en")
        assert svc.is_rtl(state) is False


# --------------------------------------------------------------------------- #
# App info + one-tap wipe
# --------------------------------------------------------------------------- #
class TestInfoAndWipe:
    def test_app_info_reports_kb_state(self, tmp_path):
        ready = AppService(
            auth_manager=AuthManager(tmp_path / "a.db"),
            index_dir=_kb_ready_dir(tmp_path),
        )
        missing = AppService(
            auth_manager=AuthManager(tmp_path / "b.db"),
            index_dir=tmp_path / "none",
        )
        assert ready.app_info().kb_ready is True
        assert missing.app_info().kb_ready is False
        assert ready.app_info().version == config.APP_VERSION

    def test_wipe_all_data_clears_db_and_session(self, svc, state):
        svc.init_session(state)
        svc.register(state, "ravi", GOOD_PIN, GOOD_PIN)
        svc.ask(state, "hi") if False else None
        svc.wipe_all_data(state)
        assert svc.auth.list_users() == []
        assert state["session"] is None
        assert state["screen"] == SCREEN_AUTH
        assert state["_haq_init"] is True
