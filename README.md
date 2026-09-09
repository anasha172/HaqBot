# HaqBot — حق بوت

**Offline, privacy-first, mobile-first multilingual legal RAG assistant for UAE Labour rights.**
Built on Intel® OpenVINO™, FAISS, SQLite, and Streamlit for the Intel® AI Global Impact Festival 2026.

HaqBot gives migrant workers in the UAE immediate, **100% on-device** guidance on their
rights under **Federal Decree-Law No. 33 of 2021** and the Wage Protection System (WPS).
No cloud. No telemetry. No network calls. Runs on low-spec Android/iOS devices and laptops.

---

## Status

| Phase | Scope | State |
| :---- | :---- | :---- |
| 1 | Project setup, mobile viewport & security config | ✅ done |
| 2 | Local offline auth (PIN) + SQLite profile store | ✅ done |
| 3 | PDF ingestion, chunking & metadata tagging | ⬜ pending |
| 4 | OpenVINO INT8 quantization + FAISS index build | ⬜ pending |
| 5 | RAG pipeline + anti-hallucination guardrails | ⬜ pending |
| 6 | Mobile-first Streamlit UI + air-gapped test suite | ⬜ pending |

---

## Quickstart

Requires **Python 3.13**.

```bash
# 1. Create an isolated environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS / Linux

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Run the test suite
pytest

# 4. Launch the app (available from Phase 6)
streamlit run app.py
```

The app binds to `localhost` only, ships with telemetry disabled
(`.streamlit/config.toml`), and performs zero outbound requests at runtime.

### Mobile / low-spec testing

* Open the local URL on a phone on the same LAN, or use Chrome DevTools
  device emulation (Pixel 5 / iPhone SE, 360–414 px width).
* Enable **Airplane Mode** on the host and confirm the full
  Auth → Chat → Settings flow still works.

---

## Repository layout

```
haqbot/
├── .streamlit/config.toml   # mobile theme, telemetry disabled, no tracebacks
├── data/
│   ├── raw/                  # source UAE legal PDFs (git-ignored by default)
│   ├── processed/            # FAISS index + chunk metadata (git-ignored)
│   └── local_user.db         # local encrypted SQLite auth/profile (git-ignored)
├── models/                   # local OpenVINO INT8 IR models (git-ignored)
├── src/
│   ├── config.py             # paths, thresholds, prompts, localisation
│   ├── auth.py               # offline PIN auth + SQLite manager   (Phase 2)
│   ├── ingestion.py          # PDF parser + article chunking       (Phase 3)
│   ├── quantization.py       # OpenVINO Optimum export utilities   (Phase 4)
│   ├── vectorstore.py        # local FAISS query engine            (Phase 4)
│   └── pipeline.py           # RAG orchestration + guardrails      (Phase 5)
├── tests/
├── app.py                    # multi-screen mobile Streamlit UI    (Phase 6)
└── requirements.txt
```

---

## Privacy & safety

* **Air-gapped:** telemetry disabled; `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`
  enforced at startup; no runtime network access.
* **Local-only auth:** 4-digit PIN, hashed before storage in on-device SQLite.
* **One-tap data purge:** wipe all local profile and session data from Settings.
* **Anti-hallucination:** retrieval below cosine similarity `0.65` bypasses the
  LLM and returns the official MOHRE referral with citations.

> Informational guidance only — not legal advice. For binding decisions contact
> MOHRE at **80084** or **www.mohre.gov.ae**.
