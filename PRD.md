# HaqBot — Mobile-First Production PRD & Execution Plan
> **Offline Multilingual Legal RAG Assistant for UAE Labour Rights (Intel® OpenVINO™)**
> **Project:** HaqBot (حق بوت) | **Intel® AI Global Impact Festival 2026**
> **Target Stack:** Intel® OpenVINO™ Toolkit, LangChain, FAISS, Streamlit / PWA (Progressive Web App), SQLite (Local Encrypted Auth)
> **Deployment Target:** Low-End Android/iOS Devices & Laptops | 100% On-Device / Air-Gapped / Zero Network Calls

---

## 1. Executive Summary & Core Mobile-First Objective

**HaqBot** ("Haq" meaning "Right" in Arabic and Urdu) is an air-gapped, privacy-first, on-device Retrieval-Augmented Generation (RAG) system designed to empower vulnerable migrant workers in the UAE with immediate, offline access to legal rights under UAE Federal Decree-Law No. 33 of 2021 and Wage Protection System (WPS) regulations.

Recognizing that target users often rely on budget, low-spec mobile smartphones (Android 8.0+ / low RAM) and may not own laptops, HaqBot is engineered as a **mobile-first, highly responsive web application / PWA**. It runs locally on modest hardware, requiring no high-end OS updates or cloud connectivity. The platform includes a complete, end-to-end mobile user journey—from local offline authentication to chat guidance and localized profile settings.

### Core Mobile & System Non-Negotiables
- **Mobile-First & Low-Spec Compatibility:** Touch-friendly UI optimized for small viewports (360px–414px width), low memory footprints, and legacy OS versions (Android 8+ / iOS 12+).
- **Full App Experience:** Complete workflow including Local Offline PIN/Login Screen, Main RAG Chatbot Interface, and Profile & Settings Screen.
- **Zero Cloud / Zero Data Leakage:** 100% on-device processing via INT8 quantized models. Zero remote logging or telemetry.
- **Claude Code Optimization:** Deterministic, modular 6-phase development plan engineered to minimize agent token usage.

---

## 2. Full Application Flow & Mobile User Experience (UX)

The mobile interface is designed as an intuitive 3-screen application flow tailored for low literacy and multi-language access:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           1. AUTH & PIN SCREEN                          │
│  - Multilingual Language Selector (Hindi / Urdu / Malayalam / Arabic)   │
│  - Simple 4-Digit Local Offline PIN Entry / Guest Quick Access          │
│  - Stored in Local Encrypted SQLite (No Remote Servers)                 │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        2. MAIN CHATBOT INTERFACE                        │
│  - Sticky Mobile Header with Offline Indicator ("100% Air-Gapped")       │
│  - Audio-Guided Voice Prompts / Large Mobile Touch Buttons             │
│  - Real-time Citation Expanders & Legal Article Extracts                │
│  - Emergency Helpline Floating Action Button (MOHRE 80084)               │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     3. USER PROFILE & SETTINGS                          │
│  - Preferred Native Language & Font Scaling (Accessibility)            │
│  - Contract Type / Basic Worker Profile (Stored Locally)               │
│  - One-Tap "Clear All Session Data" (Privacy Purge)                     │
│  - App Version & Offline Legal Knowledge Base Snapshot Timestamp         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Architecture & Mobile Tech Stack

| Component | Technology / Tool | Mobile & Performance Specification |
| :--- | :--- | :--- |
| **Front-End UI** | Streamlit (Mobile Layout) / PWA Manifest | Mobile viewport styling (`viewportwidth=device-width`), touch-optimized CSS, collapsible drawers. |
| **Local Auth & Profile** | SQLite (Local) / `st.session_state` | Local user credentials and settings stored in air-gapped local database file (`data/local_user.db`). |
| **LLM Engine** | Qwen2.5-1.5B-Instruct / Qwen2.5-3B-Instruct | Quantized to `INT8` via OpenVINO. Low RAM consumption (~1.2GB CPU RAM target). |
| **Embedding Engine** | `multilingual-e5-small` (OpenVINO IR) | Extremely fast CPU retrieval (<50ms) with full multilingual tokenization. |
| **Vector DB** | FAISS (Flat IP / CPU) | Flat index loaded into RAM for zero-latency local vector search. |
| **Parser & Ingestion** | PyMuPDF | Extracts structured legal text and metadata headers directly on device. |

---

## 4. AI Ethics, Safety Guardrails & Zero-Leakage Privacy

### 4.1 On-Device Security & Data Isolation
- **Local SQLite Encrypted Storage:** User profile info and local PIN hashed via SHA-256 stored purely on device storage.
- **One-Tap Data Purge:** Users can wipe all local logs and profile preferences instantly from the settings screen.
- **Air-Gapped Telemetry Block:** Telemetry explicitly disabled in `.streamlit/config.toml` (`browser.gatherUsageStats = false`).

### 4.2 Guardrails & Citation Enforcement
> **Strict Legal Refusal Guardrail:** If similarity score $S_{	ext{thresh}} < 0.65$, generation is skipped, outputting a standard fallback directing users to official MOHRE channels.

```python
SYSTEM_PROMPT = """You are HaqBot, an official AI Legal Assistant providing informational guidance on UAE Labour Law (Federal Decree-Law No. 33 of 2021) and Wages Protection System (WPS) regulations.

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
```

---

## 5. Phased Implementation Plan for Claude Code Execution

To prevent token exhaustion and maintain strict agent control, the workflow is split into 6 modular build phases.

### PHASE 1: Project Setup, Mobile Viewport & Security Configurations
**Objective:** Initialize workspace, pin dependencies, configure mobile viewport and offline parameters.
- [ ] Setup folder structure: `data/raw/`, `data/processed/`, `models/`, `src/`, `tests/`.
- [ ] Create `requirements.txt`: `openvino==2024.1.0`, `optimum-intel[openvino]==1.16.0`, `langchain==0.2.1`, `faiss-cpu==1.8.0`, `streamlit==1.35.0`, `pymupdf==1.24.3`.
- [ ] Configure `.streamlit/config.toml` for mobile viewport sizing, telemetry blocking, and custom color scheme.
- [ ] Create `src/config.py` with system constants, paths, and thresholds ($S_{	ext{thresh}} = 0.65$).

### PHASE 2: Local Auth System & Database Setup
**Objective:** Implement local offline authentication (PIN system) and profile database using SQLite.
- [ ] Build `src/auth.py` with SQLite integration for local PIN setup, hashing, and authentication verification.
- [ ] Create local user profile model storing language preferences and accessibility font sizes.
- [ ] Build unit tests in `tests/test_auth.py` verifying offline auth flow without network connection.

### PHASE 3: Ingestion, Chunking & Metadata Tagging
**Objective:** Cleanly parse UAE Labour Law documents and attach strict metadata headers.
- [ ] Create `src/ingestion.py` using PyMuPDF to extract text from UAE Decree-Law No. 33 and WPS PDFs.
- [ ] Apply regular expressions to capture Article numbers, Clause headers, and section topics.
- [ ] Implement `RecursiveCharacterTextSplitter` (chunk size: 450, overlap: 50) storing chunk metadata.

### PHASE 4: OpenVINO Quantization & Vector Index Construction
**Objective:** Export models to OpenVINO INT8 format and persist local FAISS index.
- [ ] Build `src/quantization.py` utilizing `optimum-cli` to export embedding model and LLM to INT8 OpenVINO IR format.
- [ ] Develop `src/vectorstore.py` to generate and persist FAISS index on disk (`data/processed/faiss_index`).
- [ ] Verify sub-100ms retrieval speed on low-power CPU in `tests/test_vectorstore.py`.

### PHASE 5: RAG Pipeline Engine & Anti-Hallucination Guardrails
**Objective:** Connect retrieval search, similarity score evaluation, and grounded OpenVINO LLM inference.
- [ ] Build `src/pipeline.py` wrapping OpenVINO model invocation via LangChain.
- [ ] Enforce similarity thresholding ($S_{	ext{thresh}} < 0.65$) to trigger instant fallback refusals when context is lacking.
- [ ] Integrate multilingual response formatting (Hindi, Urdu, Malayalam, Arabic, English).

### PHASE 6: Mobile-First Front-End App Implementation & Testing
**Objective:** Construct complete multi-screen mobile UI and execute full air-gapped test suite.
- [ ] Implement `app.py` featuring a 3-tab / page navigation bar:
  1. **Login / PIN Screen**
  2. **Chatbot Interface** (sticky header, expandable citations, touch-friendly UI)
  3. **Profile & Settings** (language picker, font scaling, data wipe button)
- [ ] Test application in OS **Airplane Mode** across mobile viewports to ensure 100% offline capability.

---

## 6. Complete Repository Structure

```
haqbot/
├── .streamlit/
│   └── config.toml             # Mobile viewport, telemetry disabled & offline theme
├── data/
│   ├── raw/                    # Primary UAE Legal PDFs
│   ├── processed/              # FAISS index files
│   └── local_user.db           # Encrypted local SQLite database for local PIN/profile
├── models/
│   ├── e5-small-ov/            # Quantized OpenVINO embedding model
│   └── qwen2.5-1.5b-ov-int8/   # Quantized OpenVINO INT8 mobile LLM
├── src/
│   ├── __init__.py
│   ├── config.py               # Path definitions & thresholds
│   ├── auth.py                 # Offline PIN authentication & SQLite manager
│   ├── ingestion.py            # PDF parser & article chunking
│   ├── quantization.py         # OpenVINO Optimum export utilities
│   ├── vectorstore.py          # Local FAISS index query engine
│   └── pipeline.py            # RAG orchestration, threshold check, prompts
├── tests/
│   ├── test_auth.py            # Local SQLite & PIN verification
│   ├── test_ingestion.py       # Article regex extraction tests
│   └── test_offline.py         # Airplane mode network isolation tests
├── app.py                      # Multi-screen Mobile-First Streamlit Interface
├── PRD.md                      # Updated Production Mobile PRD
├── requirements.txt            # Pinned lightweight dependencies
└── README.md                   # Quickstart guide & mobile setup instructions
```

---

## 7. Mobile Compliance & Hackathon Verification Checklist

| Area | Compliance Standard | Verification Method |
| :--- | :--- | :--- |
| **Mobile UX** | Touch-friendly viewports (360px+) | Test layout in Chrome DevTools mobile responsive mode (Pixel 5 / iPhone SE). |
| **Low Hardware Target** | Sub-1.5GB RAM & Low CPU usage | Quantized INT8 models (`Qwen2.5-1.5B`/`3B`) tested on dual-core CPU environment. |
| **Offline Security** | UAE PDPL & Air-Gapped Operation | Execute complete workflow in Airplane Mode with network interfaces disabled. |
| **Full Workflow** | Auth -> Chatbot -> Profile Settings | Verify seamless user transition across all 3 app screens using local SQLite state. |
