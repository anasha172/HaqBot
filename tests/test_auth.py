"""Phase 2 — offline local authentication & profile store.

All tests use a throwaway SQLite file under ``tmp_path``; nothing touches the
real ``data/local_user.db`` or the network.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys

import pytest

from src import auth, config
from src.auth import (
    AccountLockedError,
    AuthManager,
    AuthSession,
    InvalidPinError,
    PinPolicyError,
    UserExistsError,
    UsernamePolicyError,
    UserNotFoundError,
    hash_pin,
    validate_pin,
    verify_pin_hash,
)

GOOD_PIN = "8351"
OTHER_PIN = "2947"


@pytest.fixture
def auth_mgr(tmp_path):
    mgr = AuthManager(tmp_path / "local_user.db")
    try:
        yield mgr
    finally:
        mgr.close()


def _raw_profile_row(db_path) -> sqlite3.Row:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM profiles").fetchone()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
class TestValidatePin:
    @pytest.mark.parametrize("pin", ["8351", "1937", "0264", "1002"])
    def test_accepts_strong_four_digit_pins(self, pin):
        assert validate_pin(pin) == pin

    @pytest.mark.parametrize("pin", ["123", "12345", "", "83a1", "83 1", "８３５１"])
    def test_rejects_bad_format(self, pin):
        with pytest.raises(PinPolicyError):
            validate_pin(pin)

    @pytest.mark.parametrize("pin", ["0000", "1111", "9999"])
    def test_rejects_repeated_digits(self, pin):
        with pytest.raises(PinPolicyError):
            validate_pin(pin)

    @pytest.mark.parametrize("pin", ["1234", "2345", "6789", "4321", "9876"])
    def test_rejects_sequences(self, pin):
        with pytest.raises(PinPolicyError):
            validate_pin(pin)

    def test_rejects_non_string(self):
        with pytest.raises(PinPolicyError):
            validate_pin(8351)  # type: ignore[arg-type]


class TestValidateUsername:
    @pytest.mark.parametrize("name", ["ravi", "ravi.k", "worker-01", "a_b", "7ml"])
    def test_accepts_and_normalizes(self, name):
        assert auth.validate_username(name.upper()) == name

    @pytest.mark.parametrize("name", ["", "   ", "-lead", ".lead", "bad name",
                                      "toolong_" * 5, "üser", "a/b"])
    def test_rejects_bad(self, name):
        with pytest.raises(UsernamePolicyError):
            auth.validate_username(name)


class TestHashPin:
    def test_roundtrip(self):
        h, salt, iters = hash_pin(GOOD_PIN)
        assert verify_pin_hash(GOOD_PIN, h, salt, iters) is True
        assert verify_pin_hash(OTHER_PIN, h, salt, iters) is False

    def test_salt_is_random_per_call(self):
        assert hash_pin(GOOD_PIN)[1] != hash_pin(GOOD_PIN)[1]

    def test_deterministic_with_fixed_salt(self):
        salt = b"\x00" * 16
        a = hash_pin(GOOD_PIN, salt=salt)
        b = hash_pin(GOOD_PIN, salt=salt)
        assert a == b
        assert a[2] == config.PBKDF2_ITERATIONS

    def test_pin_not_recoverable_from_hash(self):
        h, _salt, _iters = hash_pin(GOOD_PIN)
        assert GOOD_PIN not in h

    def test_tampered_hash_rejected(self):
        h, salt, iters = hash_pin(GOOD_PIN)
        flipped = h[:-1] + ("1" if h[-1] != "1" else "2")
        assert verify_pin_hash(GOOD_PIN, flipped, salt, iters) is False
        assert verify_pin_hash(GOOD_PIN, h, "not-hex", iters) is False
        assert verify_pin_hash(GOOD_PIN, h, None, iters) is False  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
class TestRegister:
    def test_creates_user_and_default_profile(self, auth_mgr):
        profile = auth_mgr.register("Ravi", GOOD_PIN)
        assert profile.username == "ravi"
        assert profile.language == config.DEFAULT_LANGUAGE
        assert profile.font_scale == config.DEFAULT_FONT_SCALE
        assert auth_mgr.user_exists("ravi") is True
        assert auth_mgr.has_any_pin_user() is True

    def test_duplicate_rejected(self, auth_mgr):
        auth_mgr.register("ravi", GOOD_PIN)
        with pytest.raises(UserExistsError):
            auth_mgr.register("RAVI", OTHER_PIN)

    def test_weak_pin_rejected(self, auth_mgr):
        with pytest.raises(PinPolicyError):
            auth_mgr.register("ravi", "1234")
        assert auth_mgr.user_exists("ravi") is False

    def test_reserved_guest_name_rejected(self, auth_mgr):
        with pytest.raises(UsernamePolicyError):
            auth_mgr.register("guest", GOOD_PIN)


# --------------------------------------------------------------------------- #
# Authentication + lockout
# --------------------------------------------------------------------------- #
class TestAuthenticate:
    def test_success_returns_session(self, auth_mgr):
        auth_mgr.register("ravi", GOOD_PIN)
        session = auth_mgr.authenticate("RAVI", GOOD_PIN)
        assert isinstance(session, AuthSession)
        assert session.username == "ravi"
        assert session.is_guest is False
        assert session.profile.language == config.DEFAULT_LANGUAGE

    def test_unknown_user(self, auth_mgr):
        with pytest.raises(UserNotFoundError):
            auth_mgr.authenticate("nobody", GOOD_PIN)

    def test_wrong_pin_counts_down(self, auth_mgr):
        auth_mgr.register("ravi", GOOD_PIN)
        with pytest.raises(InvalidPinError) as e1:
            auth_mgr.authenticate("ravi", OTHER_PIN)
        assert "4 attempts left" in str(e1.value)
        with pytest.raises(InvalidPinError) as e2:
            auth_mgr.authenticate("ravi", OTHER_PIN)
        assert "3 attempts left" in str(e2.value)

    def test_locks_after_max_attempts(self, auth_mgr):
        auth_mgr.register("ravi", GOOD_PIN)
        for _ in range(config.MAX_PIN_ATTEMPTS - 1):
            with pytest.raises(InvalidPinError):
                auth_mgr.authenticate("ravi", OTHER_PIN)
        with pytest.raises(AccountLockedError):
            auth_mgr.authenticate("ravi", OTHER_PIN)
        # even the correct PIN is refused during cooldown
        with pytest.raises(AccountLockedError):
            auth_mgr.authenticate("ravi", GOOD_PIN)

    def test_cooldown_expiry_allows_login_and_resets(self, auth_mgr):
        auth_mgr.register("ravi", GOOD_PIN)
        for _ in range(config.MAX_PIN_ATTEMPTS):
            with pytest.raises((InvalidPinError, AccountLockedError)):
                auth_mgr.authenticate("ravi", OTHER_PIN)
        # force the lock into the past
        conn = auth_mgr._connection()
        conn.execute(
            "UPDATE users SET locked_until = '2000-01-01T00:00:00Z' "
            "WHERE username = 'ravi'"
        )
        conn.commit()
        session = auth_mgr.authenticate("ravi", GOOD_PIN)
        assert session.username == "ravi"
        row = auth_mgr._get_user_row("ravi")
        assert row["failed_attempts"] == 0
        assert row["locked_until"] is None


# --------------------------------------------------------------------------- #
# Guest
# --------------------------------------------------------------------------- #
class TestGuest:
    def test_login_guest_is_idempotent(self, auth_mgr):
        s1 = auth_mgr.login_guest()
        s2 = auth_mgr.login_guest()
        assert s1.is_guest and s2.is_guest
        assert s1.profile.is_guest is True
        assert s1._data_key is None
        assert auth_mgr.list_users().count("guest") == 1

    def test_guest_cannot_authenticate_with_pin(self, auth_mgr):
        auth_mgr.login_guest()
        with pytest.raises(UserNotFoundError):
            auth_mgr.authenticate("guest", GOOD_PIN)

    def test_guest_may_change_language_but_not_secrets(self, auth_mgr):
        session = auth_mgr.login_guest()
        auth_mgr.update_profile(session, language="ur")
        assert auth_mgr.get_public_profile("guest").language == "ur"
        with pytest.raises(auth.AuthError):
            auth_mgr.update_profile(session, contract_type="Limited")


# --------------------------------------------------------------------------- #
# Profile CRUD + at-rest encryption
# --------------------------------------------------------------------------- #
class TestProfile:
    def test_language_and_font_persist_across_reconnect(self, auth_mgr, tmp_path):
        auth_mgr.register("ravi", GOOD_PIN)
        session = auth_mgr.authenticate("ravi", GOOD_PIN)
        auth_mgr.update_profile(session, language="ml", font_scale="Large")
        auth_mgr.close()

        reopened = AuthManager(tmp_path / "local_user.db")
        try:
            pub = reopened.get_public_profile("ravi")
            assert (pub.language, pub.font_scale) == ("ml", "Large")
        finally:
            reopened.close()

    def test_invalid_language_or_font_rejected(self, auth_mgr):
        auth_mgr.register("ravi", GOOD_PIN)
        session = auth_mgr.authenticate("ravi", GOOD_PIN)
        with pytest.raises(auth.AuthError):
            auth_mgr.update_profile(session, language="fr")
        with pytest.raises(auth.AuthError):
            auth_mgr.update_profile(session, font_scale="Huge")

    def test_contract_type_is_encrypted_at_rest(self, auth_mgr, tmp_path):
        auth_mgr.register("ravi", GOOD_PIN)
        session = auth_mgr.authenticate("ravi", GOOD_PIN)
        auth_mgr.update_profile(
            session, contract_type="Limited", worker_sector="Construction"
        )
        assert session.profile.contract_type == "Limited"

        row = _raw_profile_row(tmp_path / "local_user.db")
        assert row["contract_type_enc"] not in (None, "Limited")
        assert "Limited" not in (row["contract_type_enc"] or "")
        assert "Construction" not in (row["worker_sector_enc"] or "")

        # a fresh sign-in decrypts them back
        auth_mgr.close()
        reopened = AuthManager(tmp_path / "local_user.db")
        try:
            s2 = reopened.authenticate("ravi", GOOD_PIN)
            assert s2.profile.contract_type == "Limited"
            assert s2.profile.worker_sector == "Construction"
        finally:
            reopened.close()

    def test_clearing_a_secret_sets_null(self, auth_mgr, tmp_path):
        auth_mgr.register("ravi", GOOD_PIN)
        session = auth_mgr.authenticate("ravi", GOOD_PIN)
        auth_mgr.update_profile(session, contract_type="Unlimited")
        auth_mgr.update_profile(session, contract_type=None)
        assert session.profile.contract_type is None
        row = _raw_profile_row(tmp_path / "local_user.db")
        assert row["contract_type_enc"] is None

    def test_get_public_profile_hides_secrets(self, auth_mgr):
        auth_mgr.register("ravi", GOOD_PIN)
        session = auth_mgr.authenticate("ravi", GOOD_PIN)
        auth_mgr.update_profile(session, contract_type="Limited")
        pub = auth_mgr.get_public_profile("ravi")
        assert pub.contract_type is None
        assert pub.language == config.DEFAULT_LANGUAGE

    def test_get_public_profile_unknown_user(self, auth_mgr):
        with pytest.raises(UserNotFoundError):
            auth_mgr.get_public_profile("ghost")

    def test_update_profile_requires_session(self, auth_mgr):
        with pytest.raises(auth.AuthError):
            auth_mgr.update_profile(None)  # type: ignore[arg-type]

    def test_update_profile_noop_when_nothing_changes(self, auth_mgr):
        auth_mgr.register("ravi", GOOD_PIN)
        session = auth_mgr.authenticate("ravi", GOOD_PIN)
        before = session.profile.updated_at
        result = auth_mgr.update_profile(session)
        assert result is session.profile
        assert result.updated_at == before


# --------------------------------------------------------------------------- #
# Change PIN
# --------------------------------------------------------------------------- #
class TestChangePin:
    def test_changes_pin_and_preserves_secret(self, auth_mgr, tmp_path):
        auth_mgr.register("ravi", GOOD_PIN)
        session = auth_mgr.authenticate("ravi", GOOD_PIN)
        auth_mgr.update_profile(session, contract_type="Limited")

        auth_mgr.change_pin("ravi", GOOD_PIN, OTHER_PIN)

        with pytest.raises(InvalidPinError):
            auth_mgr.authenticate("ravi", GOOD_PIN)
        s2 = auth_mgr.authenticate("ravi", OTHER_PIN)
        assert s2.profile.contract_type == "Limited"

    def test_wrong_old_pin_rejected(self, auth_mgr):
        auth_mgr.register("ravi", GOOD_PIN)
        with pytest.raises(InvalidPinError):
            auth_mgr.change_pin("ravi", "1092", OTHER_PIN)

    def test_weak_new_pin_rejected(self, auth_mgr):
        auth_mgr.register("ravi", GOOD_PIN)
        with pytest.raises(PinPolicyError):
            auth_mgr.change_pin("ravi", GOOD_PIN, "1111")


# --------------------------------------------------------------------------- #
# Deletion / purge
# --------------------------------------------------------------------------- #
class TestPurge:
    def test_delete_user_cascades_profile(self, auth_mgr, tmp_path):
        auth_mgr.register("ravi", GOOD_PIN)
        auth_mgr.delete_user("ravi")
        assert auth_mgr.user_exists("ravi") is False
        row = _raw_profile_row(tmp_path / "local_user.db")
        assert row is None

    def test_delete_unknown_user(self, auth_mgr):
        with pytest.raises(UserNotFoundError):
            auth_mgr.delete_user("ghost")

    def test_wipe_all_removes_file_and_reinitializes(self, auth_mgr, tmp_path):
        auth_mgr.register("ravi", GOOD_PIN)
        auth_mgr.login_guest()
        db_file = tmp_path / "local_user.db"
        assert db_file.exists()

        auth_mgr.wipe_all()

        assert db_file.exists()  # recreated, empty
        assert auth_mgr.list_users() == []
        # still usable afterwards
        auth_mgr.register("ravi", GOOD_PIN)
        assert auth_mgr.user_exists("ravi")

    def test_wipe_all_without_reinit(self, auth_mgr, tmp_path):
        auth_mgr.register("ravi", GOOD_PIN)
        auth_mgr.wipe_all(reinitialize=False)
        assert not (tmp_path / "local_user.db").exists()


# --------------------------------------------------------------------------- #
# Session hygiene
# --------------------------------------------------------------------------- #
class TestSessionHygiene:
    def test_repr_hides_data_key(self, auth_mgr):
        auth_mgr.register("ravi", GOOD_PIN)
        session = auth_mgr.authenticate("ravi", GOOD_PIN)
        text = repr(session)
        assert "ravi" in text
        assert session._data_key is not None
        assert str(session._data_key) not in text
        assert "_data_key" not in text

    def test_context_manager_closes(self, tmp_path):
        with AuthManager(tmp_path / "local_user.db") as mgr:
            mgr.register("ravi", GOOD_PIN)
        assert mgr._conn is None


# --------------------------------------------------------------------------- #
# Offline / air-gap guarantees
# --------------------------------------------------------------------------- #
@pytest.mark.offline
class TestOffline:
    def test_full_flow_with_sockets_disabled(self, tmp_path, monkeypatch):
        import socket

        def _no_network(*_a, **_k):
            raise AssertionError("network access attempted during auth flow")

        monkeypatch.setattr(socket, "socket", _no_network)
        monkeypatch.setattr(socket, "create_connection", _no_network)

        mgr = AuthManager(tmp_path / "local_user.db")
        try:
            mgr.register("ravi", GOOD_PIN)
            s = mgr.authenticate("ravi", GOOD_PIN)
            mgr.update_profile(s, language="hi", contract_type="Limited")
            mgr.change_pin("ravi", GOOD_PIN, OTHER_PIN)
            s2 = mgr.authenticate("ravi", OTHER_PIN)
            assert s2.profile.contract_type == "Limited"
            assert s2.profile.language == "hi"
            mgr.login_guest()
            mgr.wipe_all()
            assert mgr.list_users() == []
        finally:
            mgr.close()

    def test_auth_module_imports_no_network_libs(self):
        code = (
            "import sys, src.auth; "
            "bad = {'requests','httpx','urllib.request','aiohttp','openvino',"
            "'torch','transformers','streamlit'} & set(sys.modules); "
            "print(sorted(bad))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=config.BASE_DIR, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[]", result.stdout
