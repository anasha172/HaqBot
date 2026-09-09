"""HaqBot — 100% offline local authentication and profile store.

Everything here runs on-device against a single local SQLite file
(``data/local_user.db``). No network, no remote server, no telemetry.

Security model (PRD §4.1):
  * The 4-digit PIN is **never stored** — only a PBKDF2-HMAC-SHA256 digest with a
    per-user 16-byte salt and 200 000 iterations, compared in constant time.
  * Sensitive free-text profile fields (``contract_type``, ``worker_sector``) are
    Fernet-encrypted at rest with a key derived from the PIN, so they are
    unreadable if the device is seized without the PIN. Language and font-size
    preferences stay in clear text so the login screen can localise itself
    before the user authenticates.
  * ``guest`` is a PIN-less quick-access account and stores no encrypted fields.
  * Brute force is bounded: after :data:`config.MAX_PIN_ATTEMPTS` failures the
    account is locked for :data:`config.LOCKOUT_COOLDOWN_SECONDS`.
  * :meth:`AuthManager.wipe_all` deletes the database file outright — the
    one-tap privacy purge behind the Settings screen.

Every public method raises a typed :class:`AuthError`; no raw exception or
traceback is ever surfaced to the UI layer.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from cryptography.fernet import Fernet, InvalidToken

from src import config

__all__ = [
    "AuthError",
    "PinPolicyError",
    "InvalidPinError",
    "UserExistsError",
    "UserNotFoundError",
    "AccountLockedError",
    "UsernamePolicyError",
    "UserProfile",
    "AuthSession",
    "AuthManager",
    "hash_pin",
    "verify_pin_hash",
    "validate_pin",
    "normalize_username",
]

_UNSET: Final = object()
_USERNAME_RE: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$")


# =========================================================================== #
# Exceptions — all user-safe, all carry a plain message
# =========================================================================== #
class AuthError(Exception):
    """Base class for every authentication / profile error."""


class UsernamePolicyError(AuthError):
    """Username is empty or contains disallowed characters."""


class PinPolicyError(AuthError):
    """PIN fails the format or strength policy at registration / change."""


class InvalidPinError(AuthError):
    """PIN supplied at sign-in does not match the stored digest."""


class UserExistsError(AuthError):
    """A user with this name already exists."""


class UserNotFoundError(AuthError):
    """No user with this name exists."""


class AccountLockedError(AuthError):
    """Too many failed attempts; the account is in cooldown."""

    def __init__(self, seconds_remaining: int) -> None:
        self.seconds_remaining = max(0, int(seconds_remaining))
        mins = (self.seconds_remaining + 59) // 60
        super().__init__(
            f"Too many incorrect PIN attempts. Try again in about {mins} "
            f"minute{'s' if mins != 1 else ''}."
        )


# =========================================================================== #
# Pure helpers (no I/O) — independently unit-tested
# =========================================================================== #
def normalize_username(username: str) -> str:
    """Lower-case and trim a username. Does not validate."""
    return (username or "").strip().lower()


def validate_username(username: str) -> str:
    """Return the normalized username or raise :class:`UsernamePolicyError`."""
    norm = normalize_username(username)
    if not norm:
        raise UsernamePolicyError("Please enter a name.")
    if not _USERNAME_RE.match(norm):
        raise UsernamePolicyError(
            "Name may use letters, numbers, dot, dash or underscore "
            "(1–32 characters) and must start with a letter or number."
        )
    return norm


def _is_sequential(pin: str) -> bool:
    """True for strictly ascending or descending consecutive digit runs."""
    deltas = {int(b) - int(a) for a, b in zip(pin, pin[1:])}
    return deltas in ({1}, {-1})


def validate_pin(pin: str) -> str:
    """Return the PIN unchanged, or raise :class:`PinPolicyError`.

    Policy: exactly :data:`config.PIN_LENGTH` ASCII digits, not all identical,
    and not a simple ascending/descending sequence (e.g. ``1234`` / ``4321``).
    """
    if not isinstance(pin, str):
        raise PinPolicyError("PIN must be text.")
    if len(pin) != config.PIN_LENGTH or any(c not in "0123456789" for c in pin):
        raise PinPolicyError(
            f"PIN must be exactly {config.PIN_LENGTH} digits."
        )
    if len(set(pin)) == 1:
        raise PinPolicyError("PIN is too easy to guess — avoid repeated digits.")
    if _is_sequential(pin):
        raise PinPolicyError(
            "PIN is too easy to guess — avoid sequences like 1234 or 4321."
        )
    return pin


def hash_pin(
    pin: str,
    *,
    salt: bytes | None = None,
    iterations: int = config.PBKDF2_ITERATIONS,
) -> tuple[str, str, int]:
    """Hash a PIN. Returns ``(hash_hex, salt_hex, iterations)``.

    Uses PBKDF2-HMAC-SHA256. A fresh 16-byte salt is generated when not given.
    """
    if salt is None:
        salt = os.urandom(config.PIN_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iterations)
    return digest.hex(), salt.hex(), iterations


def verify_pin_hash(
    pin: str, hash_hex: str, salt_hex: str, iterations: int
) -> bool:
    """Constant-time check of a PIN against a stored digest."""
    try:
        salt = bytes.fromhex(salt_hex)
    except (ValueError, TypeError):
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256", pin.encode("utf-8"), salt, iterations
    )
    return hmac.compare_digest(candidate.hex(), hash_hex or "")


def _derive_data_key(pin: str, kdf_salt: bytes) -> bytes:
    """Derive a URL-safe base64 Fernet key from the PIN (held in memory only)."""
    raw = hashlib.pbkdf2_hmac(
        "sha256", pin.encode("utf-8"), kdf_salt, config.PBKDF2_ITERATIONS,
        dklen=32,
    )
    return base64.urlsafe_b64encode(raw)


def _encrypt_field(key: bytes, value: str) -> str:
    return Fernet(key).encrypt(value.encode("utf-8")).decode("ascii")


def _decrypt_field(key: bytes, token: str) -> str:
    try:
        return Fernet(key).decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError) as exc:  # pragma: no cover - defensive
        raise AuthError("Stored profile data could not be read.") from exc


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:  # pragma: no cover - defensive
        return None


# =========================================================================== #
# Data structures
# =========================================================================== #
@dataclass
class UserProfile:
    """Public, decrypted view of a user's settings."""

    username: str
    language: str = config.DEFAULT_LANGUAGE
    font_scale: str = config.DEFAULT_FONT_SCALE
    contract_type: str | None = None
    worker_sector: str | None = None
    is_guest: bool = False
    updated_at: str | None = None


@dataclass
class AuthSession:
    """Result of a successful sign-in.

    Holds the per-user data key **in memory only** for the duration of the
    session so profile edits can be re-encrypted. Never logged or persisted.
    """

    username: str
    profile: UserProfile
    is_guest: bool = False
    _data_key: bytes | None = field(default=None, repr=False)

    def __repr__(self) -> str:  # keep the key out of logs / tracebacks
        return (
            f"AuthSession(username={self.username!r}, is_guest={self.is_guest})"
        )


# =========================================================================== #
# Manager
# =========================================================================== #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    username        TEXT PRIMARY KEY,
    pin_hash        TEXT,
    pin_salt        TEXT,
    pin_iterations  INTEGER,
    pin_algo        TEXT,
    kdf_salt        TEXT,
    is_guest        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until    TEXT
);

CREATE TABLE IF NOT EXISTS profiles (
    username          TEXT PRIMARY KEY
                      REFERENCES users(username) ON DELETE CASCADE,
    language          TEXT NOT NULL,
    font_scale        TEXT NOT NULL,
    contract_type_enc TEXT,
    worker_sector_enc TEXT,
    updated_at        TEXT NOT NULL
);
"""


class AuthManager:
    """Offline auth + profile CRUD over a local SQLite database."""

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else config.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self.initialize()

    # -- connection ------------------------------------------------------------
    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False: Streamlit reruns may hop threads; access
            # is still single-user and serialised by the app.
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            for name, value in config.SQLITE_PRAGMAS.items():
                conn.execute(f"PRAGMA {name}={value}")
            self._conn = conn
        return self._conn

    def initialize(self) -> None:
        """Create tables if missing and stamp the schema version."""
        conn = self._connection()
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO NOTHING",
            (str(config.DB_SCHEMA_VERSION),),
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('app_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (config.APP_VERSION,),
        )
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "AuthManager":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- queries -------------------------------------------------------------
    def _get_user_row(self, username: str) -> sqlite3.Row | None:
        cur = self._connection().execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        )
        return cur.fetchone()

    def user_exists(self, username: str) -> bool:
        return self._get_user_row(normalize_username(username)) is not None

    def list_users(self) -> list[str]:
        cur = self._connection().execute(
            "SELECT username FROM users ORDER BY is_guest, created_at"
        )
        return [row["username"] for row in cur.fetchall()]

    def has_any_pin_user(self) -> bool:
        cur = self._connection().execute(
            "SELECT 1 FROM users WHERE is_guest = 0 LIMIT 1"
        )
        return cur.fetchone() is not None

    # -- registration ------------------------------------------------------
    def register(self, username: str, pin: str) -> UserProfile:
        """Create a PIN-protected user plus a default profile."""
        norm = validate_username(username)
        if norm == config.GUEST_USERNAME:
            raise UsernamePolicyError("That name is reserved.")
        validate_pin(pin)
        if self._get_user_row(norm) is not None:
            raise UserExistsError("That name is already set up on this device.")

        pin_hash, salt_hex, iterations = hash_pin(pin)
        kdf_salt = os.urandom(config.PIN_SALT_BYTES)
        now = _iso(_utcnow())
        conn = self._connection()
        conn.execute(
            "INSERT INTO users (username, pin_hash, pin_salt, pin_iterations, "
            "pin_algo, kdf_salt, is_guest, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
            (norm, pin_hash, salt_hex, iterations,
             config.PIN_HASH_ALGORITHM, kdf_salt.hex(), now),
        )
        conn.execute(
            "INSERT INTO profiles (username, language, font_scale, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (norm, config.DEFAULT_LANGUAGE, config.DEFAULT_FONT_SCALE, now),
        )
        conn.commit()
        return UserProfile(
            username=norm,
            language=config.DEFAULT_LANGUAGE,
            font_scale=config.DEFAULT_FONT_SCALE,
            updated_at=now,
        )

    # -- sign-in ---------------------------------------------------------------
    def authenticate(self, username: str, pin: str) -> AuthSession:
        """Verify a PIN and return an :class:`AuthSession` with decrypted profile.

        Raises :class:`UserNotFoundError`, :class:`AccountLockedError`, or
        :class:`InvalidPinError`.
        """
        norm = normalize_username(username)
        row = self._get_user_row(norm)
        if row is None or row["is_guest"]:
            raise UserNotFoundError("No PIN account found for that name.")

        locked_until = _parse_iso(row["locked_until"])
        now = _utcnow()
        if locked_until and now < locked_until:
            raise AccountLockedError((locked_until - now).total_seconds())

        if not verify_pin_hash(
            pin, row["pin_hash"], row["pin_salt"], row["pin_iterations"]
        ):
            self._register_failure(norm, row["failed_attempts"])
            remaining = config.MAX_PIN_ATTEMPTS - (row["failed_attempts"] + 1)
            if remaining <= 0:
                raise AccountLockedError(config.LOCKOUT_COOLDOWN_SECONDS)
            raise InvalidPinError(
                f"Incorrect PIN. {remaining} attempt"
                f"{'s' if remaining != 1 else ''} left."
            )

        # success — clear the failure counter
        conn = self._connection()
        conn.execute(
            "UPDATE users SET failed_attempts = 0, locked_until = NULL "
            "WHERE username = ?",
            (norm,),
        )
        conn.commit()

        data_key = _derive_data_key(pin, bytes.fromhex(row["kdf_salt"]))
        profile = self._load_profile(norm, data_key)
        return AuthSession(username=norm, profile=profile, _data_key=data_key)

    def _register_failure(self, username: str, prior_attempts: int) -> None:
        attempts = prior_attempts + 1
        locked_until = None
        if attempts >= config.MAX_PIN_ATTEMPTS:
            locked_until = _iso(
                _utcnow() + timedelta(seconds=config.LOCKOUT_COOLDOWN_SECONDS)
            )
        conn = self._connection()
        conn.execute(
            "UPDATE users SET failed_attempts = ?, locked_until = ? "
            "WHERE username = ?",
            (attempts, locked_until, username),
        )
        conn.commit()

    def login_guest(self) -> AuthSession:
        """Return a session for the PIN-less guest account, creating it once."""
        conn = self._connection()
        now = _iso(_utcnow())
        if self._get_user_row(config.GUEST_USERNAME) is None:
            conn.execute(
                "INSERT INTO users (username, is_guest, created_at) "
                "VALUES (?, 1, ?)",
                (config.GUEST_USERNAME, now),
            )
            conn.execute(
                "INSERT INTO profiles (username, language, font_scale, "
                "updated_at) VALUES (?, ?, ?, ?)",
                (config.GUEST_USERNAME, config.DEFAULT_LANGUAGE,
                 config.DEFAULT_FONT_SCALE, now),
            )
            conn.commit()
        profile = self._load_profile(config.GUEST_USERNAME, None)
        profile.is_guest = True
        return AuthSession(
            username=config.GUEST_USERNAME, profile=profile, is_guest=True
        )

    # -- PIN change --------------------------------------------------------
    def change_pin(self, username: str, old_pin: str, new_pin: str) -> None:
        """Re-hash the PIN and re-encrypt profile secrets under the new key."""
        session = self.authenticate(username, old_pin)  # validates old_pin
        validate_pin(new_pin)
        row = self._get_user_row(session.username)
        assert row is not None

        old_key = session._data_key
        new_pin_hash, new_salt, iterations = hash_pin(new_pin)
        new_kdf_salt = os.urandom(config.PIN_SALT_BYTES)
        new_key = _derive_data_key(new_pin, new_kdf_salt)

        conn = self._connection()
        # re-encrypt any stored secrets
        prof_row = conn.execute(
            "SELECT contract_type_enc, worker_sector_enc FROM profiles "
            "WHERE username = ?",
            (session.username,),
        ).fetchone()
        updates: dict[str, str | None] = {}
        for col in ("contract_type_enc", "worker_sector_enc"):
            token = prof_row[col] if prof_row else None
            if token and old_key is not None:
                updates[col] = _encrypt_field(
                    new_key, _decrypt_field(old_key, token)
                )
        if updates:
            set_clause = ", ".join(f"{c} = ?" for c in updates)
            conn.execute(
                f"UPDATE profiles SET {set_clause} WHERE username = ?",
                (*updates.values(), session.username),
            )
        conn.execute(
            "UPDATE users SET pin_hash = ?, pin_salt = ?, pin_iterations = ?, "
            "kdf_salt = ?, failed_attempts = 0, locked_until = NULL "
            "WHERE username = ?",
            (new_pin_hash, new_salt, iterations, new_kdf_salt.hex(),
             session.username),
        )
        conn.commit()

    # -- profile ---------------------------------------------------------------
    def _load_profile(
        self, username: str, data_key: bytes | None
    ) -> UserProfile:
        row = self._connection().execute(
            "SELECT * FROM profiles WHERE username = ?", (username,)
        ).fetchone()
        if row is None:  # pragma: no cover - defensive
            raise UserNotFoundError("Profile not found.")
        contract_type = worker_sector = None
        if data_key is not None:
            if row["contract_type_enc"]:
                contract_type = _decrypt_field(data_key, row["contract_type_enc"])
            if row["worker_sector_enc"]:
                worker_sector = _decrypt_field(data_key, row["worker_sector_enc"])
        return UserProfile(
            username=username,
            language=row["language"],
            font_scale=row["font_scale"],
            contract_type=contract_type,
            worker_sector=worker_sector,
            is_guest=bool(self._get_user_row(username)["is_guest"]),
            updated_at=row["updated_at"],
        )

    def get_public_profile(self, username: str) -> UserProfile:
        """Non-secret settings (language / font) for pre-auth localisation."""
        norm = normalize_username(username)
        row = self._connection().execute(
            "SELECT language, font_scale, updated_at FROM profiles "
            "WHERE username = ?",
            (norm,),
        ).fetchone()
        if row is None:
            raise UserNotFoundError("No profile for that name.")
        return UserProfile(
            username=norm,
            language=row["language"],
            font_scale=row["font_scale"],
            updated_at=row["updated_at"],
        )

    def update_profile(
        self,
        session: AuthSession,
        *,
        language: Any = _UNSET,
        font_scale: Any = _UNSET,
        contract_type: Any = _UNSET,
        worker_sector: Any = _UNSET,
    ) -> UserProfile:
        """Persist changed fields. Secrets require a PIN session (not guest)."""
        if not isinstance(session, AuthSession):
            raise AuthError("A valid sign-in is required to save settings.")

        sets: dict[str, Any] = {}
        if language is not _UNSET:
            if language not in config.SUPPORTED_LANGUAGES:
                raise AuthError("Unsupported language.")
            sets["language"] = language
            session.profile.language = language
        if font_scale is not _UNSET:
            if font_scale not in config.FONT_SCALE_OPTIONS:
                raise AuthError("Unsupported font size.")
            sets["font_scale"] = font_scale
            session.profile.font_scale = font_scale

        for name, value, col in (
            ("contract_type", contract_type, "contract_type_enc"),
            ("worker_sector", worker_sector, "worker_sector_enc"),
        ):
            if value is _UNSET:
                continue
            if session.is_guest or session._data_key is None:
                raise AuthError(
                    "Set up a PIN to save contract details securely."
                )
            if value in (None, ""):
                sets[col] = None
                setattr(session.profile, name, None)
            else:
                sets[col] = _encrypt_field(session._data_key, str(value))
                setattr(session.profile, name, str(value))

        if not sets:
            return session.profile

        now = _iso(_utcnow())
        sets["updated_at"] = now
        clause = ", ".join(f"{c} = ?" for c in sets)
        conn = self._connection()
        conn.execute(
            f"UPDATE profiles SET {clause} WHERE username = ?",
            (*sets.values(), session.username),
        )
        conn.commit()
        session.profile.updated_at = now
        return session.profile

    # -- deletion / purge ----------------------------------------------------
    def delete_user(self, username: str) -> None:
        norm = normalize_username(username)
        conn = self._connection()
        cur = conn.execute("DELETE FROM users WHERE username = ?", (norm,))
        conn.commit()
        if cur.rowcount == 0:
            raise UserNotFoundError("No user to remove.")

    def wipe_all(self, *, reinitialize: bool = True) -> None:
        """One-tap privacy purge: delete the entire local database file."""
        self.close()
        base = str(self.db_path)
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                Path(base + suffix).unlink()
            except FileNotFoundError:
                pass
        if reinitialize:
            self.initialize()
