# === app.py ===
# Streamlit Bulk Call Transcriber & Analyzer (Deepgram)
# ----------------------------------------------------
# Beginner-friendly app: upload many audio files, transcribe with Deepgram,
# extract special info (names, phones, emails, accounts, companies, locations,
# products, payments, commitments), basic call type guess, per-speaker sentiment,
# and export a clean Excel in your required column order.
#
# How to run locally:
# 1) Install Python 3.10+ from https://python.org
# 2) Open Terminal / Command Prompt and run:
#       pip install -r requirements.txt
# 3) Get a Deepgram API key: https://console.deepgram.com/
# 4) Run the app:
#       streamlit run app.py
# 5) In the app sidebar, paste your Deepgram API key, upload audio files, click Transcribe.

import os
import re
import io
import json
import math
import time
import string
import requests
import pandas as pd
import streamlit as st

# ---------------------- UI ----------------------
st.set_page_config(page_title="Transcribe Calls (Deepgram)", page_icon="🎧", layout="wide")
st.title("🎧 Transcribe Calls — Bulk Audio → Excel")
st.caption("Beginner-friendly: upload many recordings, get clean Excel with transcript & analytics.")

with st.sidebar:
    st.header("Settings")
    DG_KEY = st.text_input("Deepgram API Key", type="password", help="Get one at console.deepgram.com")
    diarize = st.checkbox("Speaker diarization (who's talking)", value=True)
    smart = st.checkbox("Smart formatting (punctuation, numbers)", value=True)
    language = st.text_input("Language code (optional)", value="", help="e.g. en, en-US, ur")
    st.divider()
    st.caption("Tip: Keep the app open while files process. You can download Excel at the end.")

accepted = ["mp3","wav","m4a","mp4","webm","ogg"]
files = st.file_uploader("Upload call recordings", type=accepted, accept_multiple_files=True)
start = st.button("🚀 Transcribe & Analyze", type="primary", use_container_width=True, disabled=not files)

DEEPRAM_URL = "https://api.deepgram.com/v1/listen"

# ----------------- Helper functions -----------------

def post_deepgram(api_key: str, audio_bytes: bytes, *, diarize: bool, smart: bool, language: str|None):
    if not api_key:
        raise RuntimeError("Missing Deepgram API key")
    params = {
        "diarize": "true" if diarize else "false",
        "smart_format": "true" if smart else "false",
        "punctuate": "true" if smart else "false",
        "paragraphs": "true",
    }
    if language:
        params["language"] = language
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/octet-stream"}
    r = requests.post(DEEPRAM_URL, params=params, headers=headers, data=audio_bytes, timeout=600)
    r.raise_for_status()
    return r.json()


def extract_turns(dg_json: dict):
    """Return list of dicts: [{speaker, text, start, end}]"""
    turns = []
    try:
        alt = dg_json["results"]["channels"][0]["alternatives"][0]
        paras = alt.get("paragraphs", {}).get("paragraphs", [])
        for p in paras:
            turns.append({
                "speaker": p.get("speaker", "Speaker"),
                "text": p.get("text", "").strip(),
                "start": p.get("start"),
                "end": p.get("end")
            })
    except Exception:
        pass
    return turns


def full_transcript_text(dg_json: dict) -> str:
    try:
        return dg_json["results"]["channels"][0]["alternatives"][0].get("transcript", "").strip()
    except Exception:
        return ""

# --- lightweight extractors (regex/heuristics) ---
NAME_RE = re.compile(r"\b(?:my name is|this is|i am|i'm|speaking|name:)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", re.I)
GENERIC_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b")
PHONE_RE = re.compile(r"\+?\d[\d\s\-]{7,}\d")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ACCT_RE  = re.compile(r"\b(?:acct|account|reference|ref|ticket|case)[#:\s\-]*([A-Z0-9\-]{5,})\b", re.I)
COMPANY_RE = re.compile(r"\b(?:from|at)\s+([A-Z][A-Za-z0-9&.,'\- ]{2,})\b")
CURRENCY_RE = re.compile(r"\b(?:USD|PKR|Rs\.?|\$|£|€)\s?\d{1,3}(?:[\,\d]{0,})\b", re.I)
LOCATION_RE = re.compile(r"\b(?:in|at)\s+([A-Z][A-Za-z\- ]{2,})\b")
PRODUCT_RE = re.compile(r"\b(order|plan|package|subscription|product|service)\s+([A-Za-z0-9\- ]{2,})", re.I)
COMMIT_RE = re.compile(r"\b(will|we will|i will|promise|commit|follow up|call you back|email you|refund|escalate)\b.*?\b(today|tomorrow|\d{1,2}\s?(am|pm)|\d{1,2}/\d{1,2}|next week)?", re.I)

POS_WORDS = set("thank thanks great appreciate happy resolved pleasure awesome excellent helpful".split())
NEG_WORDS = set("angry upset bad terrible frustrated cancel complaint complain issue problem not working".split())


def guess_call_type(text: str) -> str|None:
    t = text.lower()
    if any(w in t for w in ["calling you", "outreach", "special offer", "promotion", "follow up on your request"]):
        return "Outbound"
    if any(w in t for w in ["i called", "i'm calling", "can you help", "support", "my order", "my account"]):
        return "Inbound"
    return None


def simple_sentiment(text: str) -> str:
    t = text.lower().translate(str.maketrans("", "", string.punctuation))
    pos = sum(1 for w in t.split() if w in POS_WORDS)
    neg = sum(1 for w in t.split() if w in NEG_WORDS)
    if neg > pos and neg >= 2:
        return "negative"
    if pos > neg and pos >= 2:
        return "positive"
    return "neutral"


def summarize(text: str, max_words: int = 40) -> str:
    words = text.strip().split()
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")


def top_topics(text: str, k: int = 5):
    t = text.lower()
    tokens = [w.strip(string.punctuation) for w in t.split()]
    stop = set("the a an and or but if to from for with about into on at of is are was were be been being i you he she we they it this that these those my your our their not".split())
    freq = {}
    for w in tokens:
        if not w or w in stop or len(w) < 3:
            continue
        freq[w] = freq.get(w, 0) + 1
    tops = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:k]
    return [w for w,_ in tops]


def extract_special_info(text: str):
    names = set(m for m in NAME_RE.findall(text))
    if not names:
        names = set(GENERIC_NAME_RE.findall(text))
    phones = set(PHONE_RE.findall(text))
    emails = set(EMAIL_RE.findall(text))
    accounts = set(ACCT_RE.findall(text))
    companies = set(m for m in COMPANY_RE.findall(text))
    amounts = set(CURRENCY_RE.findall(text))
    locations = set(m for m in LOCATION_RE.findall(text))
    products = set((a + " " + b).strip() for a,b in PRODUCT_RE.findall(text))
    commits = set(m.group(0) for m in COMMIT_RE.finditer(text))
    return {
        "names": sorted(names),
        "phone_numbers": sorted(phones),
        "email_addresses": sorted(emails),
        "accounts_or_refs": sorted(accounts),
        "companies": sorted(companies),
        "locations": sorted(locations),
        "products_or_services": sorted(products),
        "payments_or_billing": sorted(amounts),
        "commitments_or_actions": sorted(commits),
    }


def bulletize(info: dict) -> str:
    bullets = []
    for k, v in info.items():
        if v:
            label = k.replace("_", " ").title()
            bullets.append(f"• {label}: " + "; ".join(v))
    return "\n".join(bullets)

# ----------------- Main processing -----------------
rows = []
if start and files:
    if not DG_KEY:
        st.error("Please paste your Deepgram API key in the sidebar.")
    else:
        prog = st.progress(0.0)
        status = st.empty()
        for i, f in enumerate(files, start=1):
            status.info(f"Transcribing {f.name} ({i}/{len(files)})…")
            audio = f.read()
            try:
                dg = post_deepgram(DG_KEY, audio, diarize=diarize, smart=smart, language=language or None)
            except Exception as e:
                st.error(f"Failed {f.name}: {e}")
                prog.progress(i/len(files))
                continue

            full_text = full_transcript_text(dg)
            turns = extract_turns(dg)
            # Per-speaker sentiment
            speaker_sent = {}
            for t in turns:
                spk = t.get("speaker", "Speaker")
                speaker_sent.setdefault(spk, []).append(simple_sentiment(t.get("text","")))
            agg_sent = {spk: max(set(vals), key=vals.count) for spk, vals in speaker_sent.items()} if speaker_sent else {}

            # Map to Agent/Customer if 2 speakers exist (heuristic: first speaker = Agent)
            agent_name = None
            customer_name = None
            info = extract_special_info(full_text)
            # try name attribution
            if info["names"]:
                # take the first two distinct names if present
                if len(info["names"]) >= 1:
                    agent_name = info["names"][0]
                if len(info["names"]) >= 2:
                    customer_name = info["names"][1]

            call_type = guess_call_type(full_text)
            summary = summarize(full_text, 50)
            topics = ", ".join(top_topics(full_text, 6))

            # Build speaker-labeled transcript text
            def fmt_time(t):
                if t is None:
                    return ""
                m, s = divmod(int(t), 60)
                return f"{m:02d}:{s:02d}"
            speaker_lines = [f"{t['speaker']} [{fmt_time(t['start'])}-{fmt_time(t['end'])}]: {t['text']}" for t in turns if t.get('text')]
            speaker_text = "\n".join(speaker_lines)

            # Determine sentiments for Agent/Customer
            agent_sent = None
            cust_sent = None
            if len(agg_sent) >= 2:
                # pick two most frequent speakers
                spks = list(agg_sent.keys())[:2]
                agent_sent = agg_sent.get(spks[0], "neutral")
                cust_sent = agg_sent.get(spks[1], "neutral")
            else:
                # fallback: overall
                s_overall = simple_sentiment(full_text)
                agent_sent = s_overall
                cust_sent = s_overall

            rows.append({
                "File Name": f.name,
                "Call Date & Time": "",  # Deepgram doesn't provide original call time from the file
                "Call Type": call_type if call_type else "",
                "Agent Name": agent_name or "",
                "Customer Name": customer_name or "",
                "All Special Information Extracted": bulletize(info),
                "Summary of Conversation": summary,
                "Main Topics Discussed": topics,
                "Sentiment of Agent": agent_sent,
                "Sentiment of Customer": cust_sent,
                "Full Transcript": full_text,
                "Speaker Labeled Transcript": speaker_text,
            })

            prog.progress(i/len(files))
        status.empty()

# ----------------- Results table & Excel export -----------------
if rows:
    st.subheader("Results preview")
    col_order = [
        "File Name",
        "Call Date & Time",
        "Call Type",
        "Agent Name",
        "Customer Name",
        "All Special Information Extracted",
        "Summary of Conversation",
        "Main Topics Discussed",
        "Sentiment of Agent",
        "Sentiment of Customer",
        "Full Transcript",
        "Speaker Labeled Transcript",
    ]
    df = pd.DataFrame(rows, columns=col_order)
    st.dataframe(df, use_container_width=True)

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Calls")
        ws = writer.sheets["Calls"]
        # Auto column width
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col))
            ws.set_column(i, i, min(max_len + 2, 80))
    out.seek(0)

    st.download_button(
        "⬇️ Download Excel (.xlsx)",
        data=out,
        file_name="call_center_transcripts.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.success("Done! Your Excel is ready with clean columns in the required order.")

# ----------------- Footer -----------------
st.markdown("""
**Notes**
- Deepgram provides transcription + speaker segments. Original call date/time usually comes from your phone system/CRM; you can add it later from metadata.
- Heuristics here are simple and safe for beginners. Later, we can plug an LLM for smarter summaries, topics, and names.
""")


# === requirements.txt ===
# Copy these lines into a file named requirements.txt in your repo
# (Streamlit Cloud reads this to install dependencies)
"""
streamlit==1.37.1
requests==2.32.3
pandas==2.2.2
xlsxwriter==3.2.0
"""
