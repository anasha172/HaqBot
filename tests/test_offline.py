"""Phase 6 — Airplane-Mode / air-gap isolation suite (PRD §5 Phase 6, §7).

Proves HaqBot performs zero network I/O: offline env flags are enforced,
telemetry is disabled, no networking libraries are pulled in, and the full
UI flow runs with sockets hard-disabled.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib

import pytest

from src import config

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# Static configuration
# --------------------------------------------------------------------------- #
class TestOfflineConfig:
    @pytest.mark.parametrize(
        "key", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]
    )
    def test_offline_env_flag_is_set(self, key):
        assert config.OFFLINE_ENV[key] == "1"

    def test_streamlit_telemetry_and_tracebacks_disabled(self):
        with open(config.STREAMLIT_CONFIG_PATH, "rb") as fh:
            data = tomllib.load(fh)
        assert data["browser"]["gatherUsageStats"] is False
        assert data["client"]["showErrorDetails"] == "none"
        assert data["server"]["headless"] is True

    def test_no_external_urls_in_generated_css(self):
        from src.appservice import build_css

        css = build_css(font_scale=1.2, rtl=True)
        assert "http://" not in css
        assert "https://" not in css
        assert "googleapis" not in css


# --------------------------------------------------------------------------- #
# Import hygiene — importing HaqBot modules must not load networking libs
# --------------------------------------------------------------------------- #
class TestImportHygiene:
    def test_appservice_import_sets_offline_env_and_no_network_libs(self):
        code = (
            "import os, sys; import src.appservice; "
            "print(os.environ.get('HF_HUB_OFFLINE')); "
            "bad = {'requests','httpx','aiohttp','openvino','optimum','torch',"
            "'transformers'} & set(sys.modules); "
            "print(sorted(bad))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=config.BASE_DIR, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        lines = result.stdout.strip().splitlines()
        assert lines[0] == "1"          # offline env enforced on import
        assert lines[1] == "[]"         # no networking / ML libs loaded


# --------------------------------------------------------------------------- #
# Full controller flow with sockets disabled
# --------------------------------------------------------------------------- #
class TestControllerFlowAirGapped:
    def test_register_login_ask_settings_wipe_without_sockets(self, tmp_path, monkeypatch):
        import socket

        def _blocked(*_a, **_k):
            raise AssertionError("network access attempted during app flow")

        monkeypatch.setattr(socket, "socket", _blocked)
        monkeypatch.setattr(socket, "create_connection", _blocked)

        from src.appservice import AppService
        from src.auth import AuthManager

        svc = AppService(
            auth_manager=AuthManager(tmp_path / "local_user.db"),
            pipeline_factory=None,
            index_dir=tmp_path / "no-index",
        )
        state: dict = {}
        svc.init_session(state)
        svc.register(state, "ravi", "8351", "8351")
        svc.set_language(state, "hi")
        answer = svc.ask(state, "what is end of service gratuity?")
        assert answer.used_fallback is True           # KB not built -> referral
        assert "80084" in answer.disclaimer
        svc.logout(state)
        svc.login(state, "ravi", "8351")
        assert state["language"] == "hi"
        svc.wipe_all_data(state)
        assert svc.auth.list_users() == []


# --------------------------------------------------------------------------- #
# Streamlit UI smoke test (AppTest runs fully in-process, no server/network)
# --------------------------------------------------------------------------- #
class TestUiSmoke:
    @pytest.fixture
    def app(self, tmp_path, monkeypatch):
        from streamlit.testing.v1 import AppTest

        monkeypatch.setenv("HAQBOT_DB_PATH", str(tmp_path / "local_user.db"))
        monkeypatch.setenv("HAQBOT_INDEX_DIR", str(tmp_path / "no-index"))
        at = AppTest.from_file(
            str(config.BASE_DIR / "app.py"), default_timeout=60
        )
        at.run()
        return at

    def test_app_renders_auth_screen_without_error(self, app):
        assert not app.exception
        labels = {ti.label for ti in app.text_input}
        assert config.ui_text("create_pin", "en") in labels
        assert any(b.label == config.ui_text("continue_as_guest", "en")
                   for b in app.button)

    def test_full_ui_flow_register_ask_settings(self, app):
        app.text_input(key="reg_user").set_value("demo")
        app.text_input(key="reg_pin").set_value("8351")
        app.text_input(key="reg_confirm").set_value("8351")
        [b for b in app.button
         if b.label == config.ui_text("create_account", "en")][0].click().run()
        assert not app.exception
        assert app.session_state["screen"] == "chat"

        app.chat_input[0].set_value("can my employer keep my passport?").run()
        assert not app.exception
        turn = app.session_state["chat_history"][0]
        assert turn["answer"].used_fallback is True      # no offline KB in tests
        assert "80084" in turn["answer"].disclaimer

        app.segmented_control[0].set_value(
            config.ui_text("nav_settings", "en")
        ).run()
        assert not app.exception
        assert app.session_state["screen"] == "settings"
        assert any(c.label.startswith("Yes, erase") for c in app.checkbox)

    def test_guest_path_and_helpline_fab_present(self, app):
        [b for b in app.button
         if b.label == config.ui_text("continue_as_guest", "en")][0].click().run()
        assert not app.exception
        assert app.session_state["screen"] == "chat"
        fab_html = " ".join(m.value for m in app.markdown)
        assert f'href="tel:{config.MOHRE_HELPLINE}"' in fab_html
        assert config.OFFLINE_BADGE_TEXT in fab_html
