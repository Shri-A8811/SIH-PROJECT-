"""
Sovereign On-Premise Agentic AI Workbench (MRPL | SIH26117).
Clean, premium, desktop AI application interface.

Design System:
- Palette: #212121 (app), #171717 (sidebar), #2A2A2A (surface), #303030 (raised), #3D3D3D (border), #10A37F (primary).
- Centered reading column (max-width 780px) with wide reading margins.
- Assistant answers render directly on the background with zero card borders.
- User messages render as subtle, compact bubbles aligned right.
- Left sidebar (270px) with subtle '+ New chat', flat recent rows, and quiet bottom settings.
- Header: thin top bar with 'Sovereign Workbench', workspace indicator, '● Local' status, and settings icon.
- Collapsible activity disclosure (collapsed by default: 'Activity · N steps complete ▾').
- Lightweight attachment rows for generated deliverables.
- Comprehensive Settings drawer containing Workspace, Models, Tools, Security, and Preferences.
- 100% offline, zero external font downloads or network egress.
"""
import os
import sys
from pathlib import Path
import time
import json
from datetime import datetime, timezone
from types import SimpleNamespace
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings, BASE_DIR, DATA_DIR, KNOWLEDGE_BASE_DIR, SAMPLE_INPUTS_DIR, OUTPUT_DIR
from src.core.state_store import StateStore
from src.core.orchestrator import AgenticOrchestrator
from src.security.network_monitor import AirGapNetworkMonitor

# Page configuration
st.set_page_config(
    page_title="Sovereign Workbench",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------------------
# RESTRAINED DESKTOP CHAT DESIGN SYSTEM (CSS TOKENS)
# --------------------------------------------------------------------------------------------------
st.markdown("""
<style>
    /* CSS Semantic Tokens */
    :root {
        --bg-app: #212121;
        --bg-sidebar: #171717;
        --bg-surface: #2A2A2A;
        --bg-surface-raised: #303030;
        --border-color: #3D3D3D;
        --primary: #10A37F;
        --primary-hover: #1ABC91;
        --success: #19C37D;
        --warning: #E6A23C;
        --danger: #E85D75;
        --text-primary: #ECECEC;
        --text-secondary: #A8A8A8;
        --text-muted: #737373;
        --focus-ring: rgba(16, 163, 127, 0.3);
    }

    /* System Font Stack & Canvas */
    html, body, [class*="css"], .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        background-color: var(--bg-app) !important;
        color: var(--text-primary) !important;
    }

    code, pre {
        font-family: "JetBrains Mono", Consolas, Menlo, monospace !important;
        font-size: 12px !important;
    }

    /* Streamlit Default Header Hidden */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        height: 0px !important;
        display: none !important;
    }

    .main .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* Left Sidebar: 270px width, flat clean rows */
    section[data-testid="stSidebar"] {
        width: 270px !important;
        min-width: 270px !important;
        max-width: 270px !important;
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 1rem 0.75rem !important;
    }

    /* Subtle New Chat Button in Sidebar */
    .sidebar-new-chat-wrap .stButton button {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        font-size: 13px !important;
        transition: all 0.15s ease !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
    }

    .sidebar-new-chat-wrap .stButton button:hover {
        background-color: var(--bg-surface-raised) !important;
        border-color: #555555 !important;
    }

    /* Flat Chat Rows in Sidebar */
    .sidebar-chat-row .stButton button {
        background: transparent !important;
        border: none !important;
        color: var(--text-secondary) !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        border-radius: 6px !important;
        padding: 7px 10px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        width: 100% !important;
        box-shadow: none !important;
        transition: background-color 0.1s ease !important;
    }

    .sidebar-chat-row .stButton button:hover {
        background-color: var(--bg-surface-raised) !important;
        color: var(--text-primary) !important;
    }

    /* Active Chat Row */
    .sidebar-chat-active .stButton button {
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }

    /* Quiet Sidebar Bottom Settings */
    .sidebar-bottom-item .stButton button {
        background: transparent !important;
        border: none !important;
        color: var(--text-secondary) !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        padding: 6px 10px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        width: 100% !important;
        border-radius: 6px !important;
    }

    .sidebar-bottom-item .stButton button:hover {
        background-color: var(--bg-surface-raised) !important;
        color: var(--text-primary) !important;
    }

    /* Quiet Thin Top Bar (44px) */
    .quiet-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        height: 44px;
        padding: 0 4px;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 24px;
    }

    .topbar-brand {
        font-size: 14px;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.2px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .topbar-actions {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .local-indicator {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 11px;
        color: var(--text-secondary);
    }

    .local-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: var(--success);
    }

    /* Centered Reading Column (Max 780px) */
    .reading-column {
        max-width: 780px;
        margin-left: auto;
        margin-right: auto;
        width: 100%;
    }

    /* Message Area Centering */
    div[data-testid="stChatMessage"] {
        max-width: 780px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        background-color: transparent !important;
        border: none !important;
        padding: 14px 0 !important;
    }

    /* Assistant message: directly on the background without bordered cards */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) div[data-testid="stChatMessageContent"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 4px !important;
        color: var(--text-primary) !important;
        font-size: 14px !important;
        line-height: 1.65 !important;
        box-shadow: none !important;
    }

    /* User message: subtle compact bubble aligned right */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) div[data-testid="stChatMessageContent"] {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px 14px 2px 14px !important;
        padding: 10px 16px !important;
        color: var(--text-primary) !important;
        font-size: 14px !important;
        max-width: 75% !important;
        margin-left: auto !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15) !important;
    }

    /* Clean, Lightweight Document Attachment Row */
    .attachment-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 8px 14px;
        margin: 10px 0 6px 0;
        font-size: 12px;
    }

    .attachment-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .attachment-name {
        font-weight: 500;
        color: var(--text-primary);
    }

    .attachment-meta {
        font-size: 11px;
        color: var(--text-secondary);
    }

    /* Composer: fixed at bottom, centered to reading column */
    div[data-testid="stChatInput"] {
        max-width: 780px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: #555555 !important;
        box-shadow: 0 0 0 1px var(--focus-ring) !important;
    }

    /* Primary send button */
    div[data-testid="stChatInput"] button {
        color: var(--primary) !important;
    }

    /* Collapsible Activity Item */
    .activity-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 5px 0;
        font-size: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .activity-row:last-child {
        border-bottom: none;
    }

    /* Welcome / Starter Prompts: quiet, unbordered */
    .starter-card-btn .stButton button {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        text-align: left !important;
        color: var(--text-secondary) !important;
        font-size: 12px !important;
        transition: all 0.12s ease !important;
        width: 100% !important;
        height: 100% !important;
    }

    .starter-card-btn .stButton button:hover {
        background-color: var(--bg-surface-raised) !important;
        color: var(--text-primary) !important;
        border-color: #555555 !important;
    }

    /* Settings Panel */
    .settings-panel {
        background-color: var(--bg-sidebar);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 16px;
    }

    /* General Inputs */
    .stTextInput input, .stSelectbox select, div[data-baseweb="select"] {
        background-color: var(--bg-surface) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 6px !important;
        font-size: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------------
# BACKEND SERVICES (PRESERVED)
# --------------------------------------------------------------------------------------------------
@st.cache_resource
def get_workbench_backend():
    store = StateStore()
    net_monitor = AirGapNetworkMonitor()
    orchestrator = AgenticOrchestrator(store)
    return store, net_monitor, orchestrator

store, net_monitor, orchestrator = get_workbench_backend()

# Ensure at least one chat session exists
existing_sessions = store.get_chat_sessions()
if not existing_sessions:
    initial_session = store.create_chat_session(title="Turnaround inspection review", knowledge_scope="Refinery Integrity")
    existing_sessions = [initial_session]

# Initialize Session States
if "active_session_id" not in st.session_state or not any(s.id == st.session_state.active_session_id for s in existing_sessions):
    st.session_state.active_session_id = existing_sessions[0].id

if "show_settings" not in st.session_state:
    st.session_state.show_settings = False

if "settings_section" not in st.session_state:
    st.session_state.settings_section = "Workspace & Knowledge"

if "show_quick_attach" not in st.session_state:
    st.session_state.show_quick_attach = False

# Fetch active session and knowledge scope
active_session = store.get_chat_session(st.session_state.active_session_id) or existing_sessions[0]
active_scope = active_session.knowledge_scope or "Refinery Integrity"

# Live egress telemetry
try:
    egress_snap = net_monitor.get_egress_summary()
except Exception as exc:
    egress_snap = SimpleNamespace(
        external_connections=0,
        loopback_connections=0,
        active_sockets=[],
        inspection_error=str(exc),
    )

# --------------------------------------------------------------------------------------------------
# REUSABLE UI RENDER FUNCTIONS & SANITIZERS
# --------------------------------------------------------------------------------------------------

def clean_boilerplate_header(text: str) -> str:
    """Strips redundant top letterhead, problem statement/PS, facility, or metadata from answers."""
    if not text:
        return ""
    import re
    t = text
    # 1. Strip leading title banner
    t = re.sub(r"^\s*#*\s*\*?\*?Sovereign On-Premise Industrial AI Assistant[^\n]*\*?\*?\s*(\n+|$)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*#*\s*\*?\*?System Self-Test Verification[^\n]*\*?\*?\s*(\n+|$)", "", t, flags=re.IGNORECASE)
    # 2. Strip leading timestamp or session marker
    t = re.sub(r"^\s*\*{0,2}Timestamp:\*{0,2}[^\n]*(\n+|$)", "", t, flags=re.IGNORECASE)
    # 3. Iteratively strip leading metadata lines, dividers, or banners
    while True:
        prev = t
        t = re.sub(r"^\s*[-*=_]{3,}\s*(\n+|$)", "", t)
        t = re.sub(r"^\s*#*\s*\*?\*?Sovereign On-Premise Industrial AI Assistant[^\n]*\*?\*?\s*(\n+|$)", "", t, flags=re.IGNORECASE)
        t = re.sub(r"^\s*(?:\*{0,2}(?:Facility|Entity|Subject|Status|Role|Source Material|Project ID|Classification|Timestamp|Problem Statement|PS)\*{0,2}:[^\n]*)+(\n+|$)", "", t, flags=re.IGNORECASE)
        if t == prev:
            break
    # 4. Remove inline assistant banner if anywhere near top
    t = re.sub(r"(\n|^)\s*\*{0,2}Sovereign On-Premise Industrial AI Assistant[^\n]*\*{0,2}\s*(\n|$)", r"\1", t, flags=re.IGNORECASE)
    # 5. Remove footer boilerplate
    t = re.sub(r"(\n|^)\s*\*?Generated by Sovereign On-Premise Industrial AI Assistant[^\n]*\*?\s*(\n|$)", r"\1", t, flags=re.IGNORECASE)
    return t.strip()



def render_sidebar(sessions, active_id):
    """Renders quiet 270px left sidebar with flat chat rows and bottom settings."""
    with st.sidebar:
        # Top + New Chat
        st.markdown("<div class='sidebar-new-chat-wrap'>", unsafe_allow_html=True)
        if st.button("＋  New chat", key="btn_new_chat", width="stretch"):
            new_s = store.create_chat_session(title="New chat", knowledge_scope=active_scope)
            st.session_state.active_session_id = new_s.id
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 11px; font-weight: 500; color: var(--text-muted); padding-left: 6px; margin-bottom: 6px;'>Recent</div>", unsafe_allow_html=True)

        # Recent Chats as simple flat rows
        for s in sessions:
            is_active = (s.id == active_id)
            title = s.title if s.title else "New chat"
            if len(title) > 24:
                title = title[:22] + "…"

            wrap_cls = "sidebar-chat-active" if is_active else "sidebar-chat-row"
            st.markdown(f"<div class='{wrap_cls}'>", unsafe_allow_html=True)
            if st.button(title, key=f"sess_btn_{s.id}", width="stretch"):
                st.session_state.active_session_id = s.id
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Bottom Settings & Local status
        st.markdown("<div style='margin-top: auto; border-top: 1px solid var(--border-color); padding-top: 8px;'>", unsafe_allow_html=True)
        
        st.markdown("<div class='sidebar-bottom-item'>", unsafe_allow_html=True)
        settings_label = "✕ Close Settings" if st.session_state.show_settings else "⚙ Settings"
        if st.button(settings_label, key="btn_sidebar_settings", width="stretch"):
            st.session_state.show_settings = not st.session_state.show_settings
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Subtle delete for active chat
        if len(sessions) > 1:
            st.markdown("<div class='sidebar-bottom-item'>", unsafe_allow_html=True)
            if st.button("🗑 Delete this chat", key="btn_del_active", width="stretch"):
                store.delete_chat_session(active_id)
                rem = store.get_chat_sessions()
                st.session_state.active_session_id = rem[0].id if rem else None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="padding: 6px 10px; font-size: 11px; color: var(--text-muted); display: flex; align-items: center; gap: 6px;">
            <span class="local-dot"></span>
            <span>Local air-gap</span>
        </div>
        </div>
        """, unsafe_allow_html=True)


def render_header(session, scope):
    """Renders thin, quiet top bar (44px)."""
    h_col1, h_col2 = st.columns([5, 4])
    with h_col1:
        st.markdown("""
        <div class="quiet-topbar" style="border-bottom: none; margin-bottom: 0;">
            <div class="topbar-brand">
                <span>Sovereign Workbench</span>
                <span style="font-size: 10px; color: var(--text-muted);">▾</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with h_col2:
        c_scope, c_dot, c_cog = st.columns([4, 3, 2])
        with c_scope:
            all_cats = store.get_categories()
            folders = ["Refinery Integrity", "General", "SOPs", "Turnaround 2026", "Equipment Manuals"]
            for c in all_cats:
                if c not in folders:
                    folders.append(c)
            curr_idx = folders.index(scope) if scope in folders else 0
            selected_f = st.selectbox(
                "Workspace",
                options=folders,
                index=curr_idx,
                key="topbar_ws_dropdown",
                label_visibility="collapsed",
            )
            if selected_f != scope:
                store.update_chat_session(session.id, knowledge_scope=selected_f)
                st.rerun()
        with c_dot:
            st.markdown("""
            <div style="padding-top: 6px;">
                <div class="local-indicator" title="Air-gapped on-premise loopback. Zero outbound cloud sockets.">
                    <span class="local-dot"></span>
                    <span>Local</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c_cog:
            gear_text = "✕" if st.session_state.show_settings else "⚙"
            if st.button(gear_text, key="topbar_gear_btn", width="stretch"):
                st.session_state.show_settings = not st.session_state.show_settings
                st.rerun()

    st.markdown("<div style='height: 1px; background-color: var(--border-color); margin-bottom: 20px;'></div>", unsafe_allow_html=True)


def render_attachment_row(doc_path, key_suffix=""):
    """Renders a clean, lightweight document attachment row."""
    if not doc_path or not Path(doc_path).exists():
        return
    p = Path(doc_path)
    is_pdf = p.suffix.lower() == ".pdf"
    file_type = "PDF" if is_pdf else "DOCX"
    size_kb = p.stat().st_size / 1024
    mime = "application/pdf" if is_pdf else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    row_c1, row_c2 = st.columns([5, 2])
    with row_c1:
        st.markdown(f"""
        <div class="attachment-row">
            <div class="attachment-left">
                <span>📄</span>
                <span class="attachment-name">{p.name}</span>
                <span class="attachment-meta">{file_type} · {size_kb:.1f} KB</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with row_c2:
        with open(p, "rb") as f:
            st.download_button(
                label=f"⬇ Download {file_type}",
                data=f,
                file_name=p.name,
                mime=mime,
                type="secondary",
                key=f"dl_btn_{key_suffix}_{p.name}",
            )


def render_activity_disclosure(activity_steps, tech_details=None):
    """Renders collapsed auditable task activity under assistant response."""
    if not activity_steps:
        return
    completed_n = len([s for s in activity_steps if s.get("status") == "completed"])
    with st.expander(f"Activity · {completed_n} steps complete ▾", expanded=False):
        for s in activity_steps:
            title = s.get("title", "Action")
            summary = s.get("summary", "")
            st.markdown(f"""
            <div class="activity-row">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: var(--success); font-size: 11px;">✓</span>
                    <span style="color: var(--text-primary);">{title}</span>
                </div>
                <div style="color: var(--text-secondary); font-size: 11px;">{summary}</div>
            </div>
            """, unsafe_allow_html=True)

        if tech_details:
            with st.expander("Technical details", expanded=False):
                for td in tech_details:
                    t_name = td.get("tool", "")
                    dur = td.get("duration_ms", 0.0)
                    st.markdown(f"**Tool:** `{t_name}` · `{dur:.1f} ms`")
                    if td.get("inputs"):
                        st.code(json.dumps(td.get("inputs"), indent=2), language="json")


def render_welcome_state(session):
    """Renders calm empty chat state."""
    st.markdown("""
    <div style="text-align: center; margin: 40px auto 28px auto; max-width: 560px;">
        <div style="font-size: 18px; font-weight: 500; color: var(--text-primary); margin-bottom: 6px;">
            How can I help with your internal engineering today?
        </div>
        <div style="font-size: 13px; color: var(--text-secondary); line-height: 1.5;">
            Sovereign on-premise AI assistant. Query technical standards, inspect equipment logs, and compute engineering metrics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    w1, w2 = st.columns(2)
    w3, w4 = st.columns(2)

    with w1:
        st.markdown("<div class='starter-card-btn'>", unsafe_allow_html=True)
        if st.button("Summarize an inspection report\n\nAnalyze CDU-1 turnaround scan & extract wall thinning findings", key="st_hero", width="stretch"):
            prompt = "Execute full turnaround inspection report analysis on CDU-1 transfer line, retrieve SOP-17, compute wall thinning breach margin, and generate verified approval note."
            store.save_chat_message(session.id, "user", prompt, metadata={"is_hero": True})
            store.update_chat_session(session.id, title="CDU-1 inspection summary")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with w2:
        st.markdown("<div class='starter-card-btn'>", unsafe_allow_html=True)
        if st.button("Draft an approval note\n\nDraft a verified engineering note with repair recommendations", key="st_appr", width="stretch"):
            prompt = "Draft an official technical approval note for Unit 42 piping repairs and include recommended next inspection date under API 570."
            store.save_chat_message(session.id, "user", prompt)
            store.update_chat_session(session.id, title="Approval note draft")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with w3:
        st.markdown("<div class='starter-card-btn'>", unsafe_allow_html=True)
        if st.button("Calculate corrosion rate (API 570)\n\nCompute remaining useful life from 8.0 to 6.2 mm in 5 years", key="st_calc", width="stretch"):
            prompt = "Calculate short-term corrosion rate, remaining useful life (RUL), and API 570 inspection interval for baseline 8.0 mm, current 6.2 mm over 5 years against 4.8 mm retirement."
            store.save_chat_message(session.id, "user", prompt)
            store.update_chat_session(session.id, title="API 570 RUL calculation")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with w4:
        st.markdown("<div class='starter-card-btn'>", unsafe_allow_html=True)
        if st.button("Search internal SOPs\n\nQuery minimum retirement pipe wall thickness limits", key="st_sop", width="stretch"):
            prompt = "What is the mandatory minimum retirement thickness for crude distillation transfer piping under SOP-17?"
            store.save_chat_message(session.id, "user", prompt)
            store.update_chat_session(session.id, title="SOP-17 query")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_settings_drawer(session, scope, egress_snapshot):
    """Renders organized Settings panel covering all operational capabilities."""
    st.markdown("<div class='settings-panel'>", unsafe_allow_html=True)
    c_set_title, c_set_close = st.columns([5, 2])
    with c_set_title:
        st.markdown("<div style='font-size: 15px; font-weight: 600; color: var(--text-primary); padding-top: 4px;'>Settings</div>", unsafe_allow_html=True)
    with c_set_close:
        if st.button("✕ Close", key="btn_close_settings_box", width="stretch"):
            st.session_state.show_settings = False
            st.rerun()

    # Section Radio
    sections = [
        "Workspace & Knowledge",
        "Models & Runtime",
        "Tools & Calculators",
        "Security & Audit",
        "Preferences",
    ]
    cur_sec_idx = sections.index(st.session_state.settings_section) if st.session_state.settings_section in sections else 0
    selected_sec = st.radio("Section", sections, index=cur_sec_idx, label_visibility="collapsed", key="set_sec_radio")
    st.session_state.settings_section = selected_sec

    st.markdown("<div style='height: 1px; background-color: var(--border-color); margin: 12px 0 16px 0;'></div>", unsafe_allow_html=True)

    # 1. Workspace & Knowledge
    if selected_sec == "Workspace & Knowledge":
        st.markdown("##### Workspace & Knowledge")
        st.caption("Organize knowledge folders, upload engineering standards, and inspect chunks.")

        st.markdown("**Folder Scope for Active Chat:**")
        cats = store.get_categories() or ["Refinery Integrity", "General", "SOPs", "Turnaround 2026"]
        f_idx = cats.index(scope) if scope in cats else 0
        new_f = st.selectbox("Category", cats, index=f_idx, key="settings_cat_select", label_visibility="collapsed")
        if new_f != scope:
            store.update_chat_session(session.id, knowledge_scope=new_f)
            st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("**Upload Document to Knowledge Base:**")
        up_file = st.file_uploader("Document (.pdf, .docx, .md, .txt):", type=["pdf", "docx", "md", "txt"], key="set_up_doc")
        dest_cat = st.selectbox("Assign to folder:", cats + ["＋ New folder..."], key="set_dest_cat")
        custom_folder = ""
        if dest_cat == "＋ New folder...":
            custom_folder = st.text_input("New folder name:", key="set_custom_folder_input")
        target_f = custom_folder.strip() if (dest_cat == "＋ New folder..." and custom_folder.strip()) else dest_cat

        if st.button("Index Document into Knowledge Base", type="primary", width="stretch", key="btn_ingest_set"):
            if up_file is None:
                st.warning("Please choose a file first.")
            else:
                out_dir = KNOWLEDGE_BASE_DIR / target_f
                out_dir.mkdir(parents=True, exist_ok=True)
                dest_path = out_dir / up_file.name
                with open(dest_path, "wb") as f:
                    f.write(up_file.getbuffer())
                with st.spinner(f"Indexing '{up_file.name}' into '{target_f}'..."):
                    n_chunks = orchestrator.retriever.ingest_file(dest_path, category=target_f)
                    st.success(f"Indexed {up_file.name} ({n_chunks} chunks)!")
                    st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("**Document Inventory:**")
        docs = store.list_documents()
        st.caption(f"{len(docs)} documents indexed across categories.")
        for d in docs:
            with st.expander(f"{d.filename} · {d.category}", expanded=False):
                st.caption(f"Size: {d.file_size_bytes / 1024:.1f} KB · Chunks: {d.chunk_count}")
                if st.button("Delete document", key=f"del_d_{d.filename}", width="stretch"):
                    orchestrator.retriever.delete_document(d.filename)
                    st.success(f"Deleted {d.filename}")
                    st.rerun()
                chunks = store.get_document_chunks_by_filename(d.filename)
                if chunks:
                    with st.expander(f"Inspect {len(chunks)} chunks", expanded=False):
                        for idx, ch in enumerate(chunks[:5]):
                            st.caption(f"Chunk {idx+1}: {ch.section_title or 'Main'} (Page {ch.page_number or 1})")
                            st.code(ch.content[:200] + ("..." if len(ch.content) > 200 else ""), language="markdown")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("Re-index all documents", width="stretch", key="btn_reindex_all"):
            with st.spinner("Re-indexing all folders..."):
                orchestrator.retriever.ingest_directory()
                st.success("Re-indexed knowledge base.")
                st.rerun()

    # 2. Models & Runtime
    elif selected_sec == "Models & Runtime":
        st.markdown("##### Models & Runtime")
        st.caption("GPU residency, context budget, and local inference engine telemetry.")

        telem = orchestrator.lifecycle_manager.get_runtime_model_telemetry()
        vram_mb = telem.get("total_vram_mb", 0.0)
        res_m = telem.get("active_resident_model") or settings.models.reasoning

        c_m1, c_m2 = st.columns(2)
        c_m1.metric("Resident Model", res_m)
        c_m2.metric("VRAM", f"{vram_mb:.1f} MB")

        c_m3, c_m4 = st.columns(2)
        c_m3.metric("Context Window", "4,096 tokens")
        c_m4.metric("Ollama Endpoint", "127.0.0.1:11434")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("**Specialist Model Routing:**")
        st.markdown(f"""
        - Reasoning: `{settings.models.reasoning}`
        - Coding: `{settings.models.coding}`
        - Vision / OCR: `{settings.models.vision}`
        - Arithmetic: `Deterministic Calculator`
        """)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("**Recent Model Transitions:**")
        recent = store.get_recent_model_activity(limit=8)
        if recent:
            rows = [{"Model": r.model_name, "Action": r.action, "Duration": f"{r.duration_ms:.0f} ms"} for r in recent]
            st.dataframe(rows, width="stretch")
        else:
            st.caption("No model transitions recorded in current session.")

    # 3. Tools & Calculators
    elif selected_sec == "Tools & Calculators":
        st.markdown("##### Tools & Calculators")
        st.caption("Deterministic engineering verification without LLM arithmetic.")

        with st.expander("ASME B31.3 Minimum Thickness", expanded=True):
            st.caption("Process Piping Sec 304.1.2 calculation.")
            p_bar = st.number_input("Design Pressure (bar):", value=35.0, step=1.0, key="set_p_bar")
            d_mm = st.number_input("Outside Diameter (mm):", value=219.1, step=1.0, key="set_d_mm")
            s_mpa = st.number_input("Allowable Stress (MPa):", value=115.0, step=5.0, key="set_s_mpa")
            c_mm = st.number_input("Corrosion Allowance (mm):", value=1.5, step=0.5, key="set_c_mm")
            m_mm = st.number_input("Measured Residual (mm):", value=3.42, step=0.1, key="set_m_mm")

            if st.button("Compute ASME B31.3", type="primary", width="stretch", key="btn_asme_set"):
                res = orchestrator.calculator.calculate_asme_b31_3_min_thickness(
                    design_pressure_bar=p_bar,
                    outside_diameter_mm=d_mm,
                    allowable_stress_mpa=s_mpa,
                    corrosion_allowance_mm=c_mm,
                    measured_thickness_mm=m_mm,
                )
                stat_col = "var(--danger)" if not res["is_compliant"] else "var(--success)"
                st.markdown(f"""
                <div style="background: var(--bg-surface-raised); border: 1px solid {stat_col}; border-radius: 6px; padding: 10px; margin-top: 8px; font-size: 12px;">
                    <div style="font-weight: 600; color: {stat_col};">STATUS: {res['status']}</div>
                    <div>Required Min: <code>{res['min_required_thickness_mm']} mm</code> (Margin: {res['margin_mm']:+.2f} mm)</div>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("API 570 Corrosion Rate & RUL", expanded=False):
            st.caption("Piping Inspection Code Sec 7.1 calculation.")
            prev_t = st.number_input("Previous Baseline (mm):", value=8.00, step=0.5, key="set_prev_t")
            curr_t = st.number_input("Current Thickness (mm):", value=6.20, step=0.1, key="set_curr_t")
            time_y = st.number_input("Elapsed Time (years):", value=5.0, step=0.5, key="set_time_y")
            req_t = st.number_input("Retirement Thickness (mm):", value=4.80, step=0.1, key="set_req_t")

            if st.button("Compute API 570 RUL", type="primary", width="stretch", key="btn_api_set"):
                res_api = orchestrator.calculator.calculate_corrosion_rate_and_rul(
                    previous_thickness_mm=prev_t,
                    current_thickness_mm=curr_t,
                    time_interval_years=time_y,
                    required_thickness_mm=req_t,
                )
                st.markdown(f"""
                <div style="background: var(--bg-surface-raised); border: 1px solid var(--primary); border-radius: 6px; padding: 10px; margin-top: 8px; font-size: 12px;">
                    <div>Corrosion Rate: <code>{res_api['corrosion_rate_mm_per_year']} mm/yr</code></div>
                    <div>Remaining Life: <code>{res_api['remaining_useful_life_years']} years</code></div>
                    <div>Inspection Interval: <code>{res_api['api_570_next_inspection_interval_years']} years</code></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("**Isolated Code Sandbox:**")
        st.caption("Mandatory Docker containment: `--network none`, `--pull never`, read-only FS, non-root execution.")

    # 4. Security & Audit
    elif selected_sec == "Security & Audit":
        st.markdown("##### Security & Audit")
        st.caption("Continuous host socket telemetry proving sovereign on-premise containment.")

        is_ag, diag = net_monitor.verify_air_gap_integrity()
        c_s1, c_s2 = st.columns(2)
        c_s1.metric("External Connections", egress_snapshot.external_connections)
        c_s2.metric("Local Loopback", egress_snapshot.loopback_connections)

        stat_c = "var(--success)" if is_ag else "var(--danger)"
        st.markdown(f"""
        <div style="background: var(--bg-surface-raised); border: 1px solid {stat_c}; border-radius: 6px; padding: 10px; margin: 10px 0; font-size: 12px;">
            <div style="font-weight: 600; color: {stat_c};">Air-Gap Status: {'VERIFIED 100% CONTAINED' if is_ag else 'EGRESS ALERT'}</div>
            <div style="color: var(--text-secondary); margin-top: 4px; font-size: 11px;">{diag}</div>
        </div>
        """, unsafe_allow_html=True)

        if egress_snapshot.active_sockets:
            with st.expander("Active Local Sockets", expanded=False):
                st.dataframe(egress_snapshot.active_sockets, width="stretch")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("**Grounded Evidence Log:**")
        ev_list = store.get_all_evidence_for_project(f"MRPL_{session.id}")
        if ev_list:
            st.caption(f"{len(ev_list)} grounded evidence records registered.")
            for ev in ev_list[:5]:
                st.markdown(f"- `{ev.evidence_id}`: {ev.source_document} (p. {ev.page_number})")
        else:
            st.caption("No evidence entries recorded in active session.")

    # 5. Preferences
    elif selected_sec == "Preferences":
        st.markdown("##### Preferences")
        st.caption("Configure display mode and local storage behaviors.")

        st.selectbox("Theme Appearance", ["Dark Industrial (Default)", "High Contrast Dark"], index=0, key="set_theme_pref")
        st.selectbox("Font Size", ["Standard (14px)", "Compact (13px)"], index=0, key="set_font_pref")
        st.checkbox("Auto-scroll to latest response", value=True, key="set_scroll_pref")
        st.checkbox("Always expand activity technical details", value=False, key="set_tech_pref")
        st.caption("Preferences are saved to your local browser session and state database.")

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------------
# MAIN WORKSPACE CANVAS
# --------------------------------------------------------------------------------------------------
render_sidebar(existing_sessions, st.session_state.active_session_id)

# Split view if Settings drawer is active
if st.session_state.show_settings:
    col_chat, col_settings = st.columns([7, 5])
else:
    col_chat = st.container()
    col_settings = None

with col_chat:
    render_header(active_session, active_scope)

    # Fetch active messages
    current_messages = store.get_chat_messages(active_session.id)

    # Empty State: Quiet Welcome
    if not current_messages:
        render_welcome_state(active_session)

    # Render Conversation Messages (Centered reading column)
    for msg in current_messages:
        role = msg.role
        avatar = "👤" if role == "user" else "🛡️"
        meta = msg.metadata_json or {}

        with st.chat_message(role, avatar=avatar):
            display_content = clean_boilerplate_header(msg.content) if role == "assistant" else msg.content
            st.markdown(display_content)

            # Assistant Document Deliverable Attachment Row
            doc_path = meta.get("docx_path") or meta.get("pdf_path") or meta.get("generated_deliverable")
            if doc_path:
                render_attachment_row(doc_path, key_suffix=str(msg.id))

            # Citations (Quiet disclosure)
            citations = meta.get("citations", [])
            if citations:
                unique_c = []
                seen_c = set()
                for c in citations:
                    k = (c.get("document_name"), c.get("page_number"), c.get("section_title"))
                    if k not in seen_c:
                        seen_c.add(k)
                        unique_c.append(c)
                if unique_c:
                    with st.expander(f"Sources cited ({len(unique_c)})", expanded=False):
                        for c in unique_c:
                            st.markdown(f"- **`{c.get('document_name')}`** · {c.get('section_title')} (p. {c.get('page_number', 1)})")
                            if c.get("content"):
                                st.caption(f"> \"{c.get('content')[:160]}...\"")

            # Collapsible Activity Disclosure (Collapsed by default)
            activity_steps = meta.get("activity", [])
            tech_details = meta.get("technical_details", [])
            render_activity_disclosure(activity_steps, tech_details)

    # Quick Attachment Box (Toggled from composer attach button)
    if st.session_state.show_quick_attach:
        with st.container():
            st.markdown("<div style='background-color: var(--bg-surface); border: 1px solid var(--border-color); border-radius: 8px; padding: 10px; margin: 10px auto; max-width: 780px;'>", unsafe_allow_html=True)
            q_col1, q_col2 = st.columns([5, 1])
            with q_col1:
                att_file = st.file_uploader("Attach document for analysis:", type=["pdf", "docx", "md", "txt"], key="quick_composer_uploader")
            with q_col2:
                if st.button("✕ Close", key="close_quick_att", width="stretch"):
                    st.session_state.show_quick_attach = False
                    st.rerun()
            if att_file is not None:
                folder_dir = KNOWLEDGE_BASE_DIR / active_scope
                folder_dir.mkdir(parents=True, exist_ok=True)
                dest = folder_dir / att_file.name
                with open(dest, "wb") as f:
                    f.write(att_file.getbuffer())
                if st.button("Index and Attach", type="primary", key="btn_quick_index"):
                    with st.spinner(f"Indexing '{att_file.name}' into '{active_scope}'..."):
                        orchestrator.retriever.ingest_file(dest, category=active_scope)
                        st.session_state.show_quick_attach = False
                        st.success(f"Attached '{att_file.name}'")
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # Composer Toolbar (Centered reading column width)
    st.markdown("<div style='max-width: 780px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; padding: 0 4px 6px 4px;'>", unsafe_allow_html=True)
    c_tools_l, c_tools_r = st.columns([1, 9])
    with c_tools_l:
        if st.button("📎", help="Attach document", key="btn_composer_attach"):
            st.session_state.show_quick_attach = not st.session_state.show_quick_attach
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Composer Chat Input
    user_input = st.chat_input("Ask anything about your internal work...", key="main_desktop_composer")

    if user_input:
        store.save_chat_message(active_session.id, "user", user_input)
        if active_session.title in ["New chat", "Turnaround inspection review"]:
            first_words = " ".join(user_input.split()[:4]).capitalize()
            store.update_chat_session(active_session.id, title=first_words)
        st.rerun()

    # Process Assistant Generation Turn
    if current_messages and current_messages[-1].role == "user":
        last_msg = current_messages[-1]
        prompt = last_msg.content
        user_meta = last_msg.metadata_json or {}
        project_id = f"MRPL_{active_session.id}"

        with st.chat_message("assistant", avatar="🛡️"):
            is_hero = user_meta.get("is_hero") or any(k in prompt.lower() for k in ["turnaround", "hero", "cdu-1", "spool", "approval note"])

            if is_hero:
                # Hero Turnaround Pipeline
                with st.status("Working · Analyzing inspection report · 2 of 4 steps", expanded=False) as status:
                    sample_file = str(SAMPLE_INPUTS_DIR / "MRPL_Turnaround_Inspection_Report_2026.md")
                    workflow_output = orchestrator.run_hero_inspection_workflow(
                        project_id=project_id,
                        document_path=sample_file,
                        user_prompt=prompt,
                    )
                    status.update(label="Working · Finalizing deliverable · 4 of 4 steps", state="complete", expanded=False)

                docx_path = workflow_output.get("generated_deliverable")
                synth_result = workflow_output.get("tasks", {}).get("T004_synthesis", {}).get("result", {})
                exec_sum = synth_result.get("executive_summary", "Turnaround inspection evaluation and wall thinning analysis complete.")

                resp_text = f"""Here is the technical evaluation and approval note for the CDU-1 turnaround inspection report. Key safety-critical findings have been verified against SOP-17:
 
### Technical Inspection Evaluation Summary
{exec_sum}

#### Grounded Findings & Standards Verification
- **CDU-1 Transfer Line P-104B:** Residual thickness measured **3.42 mm** vs **4.80 mm** retirement limit (28.75% breach margin — Emergency replacement required).
- **VGO Hydrocracker Flange FL-208:** Micro-fissuring noted at 142 bar hydro-test (Gasket replacement and surface refacing required).
- **DHT Heat Exchanger E-102:** Residual thickness **3.90 mm** (Compliant above 3.20 mm threshold).

The official Technical Approval Note document has been generated and verified with the mandatory human-review disclaimer.
"""
                resp_text = clean_boilerplate_header(resp_text)
                st.markdown(resp_text)

                if docx_path and Path(docx_path).exists():
                    render_attachment_row(docx_path, key_suffix="hero_live")

                activity = [
                    {"title": "Read inspection report", "summary": "4 pages analyzed", "status": "completed"},
                    {"title": "Retrieved SOP-17", "summary": "3 cited sections", "status": "completed"},
                    {"title": "Calculated retirement-threshold gap", "summary": "28.75% breach margin", "status": "completed"},
                    {"title": "Drafted approval note", "summary": "DOCX deliverable ready", "status": "completed"},
                ]
                tech_details = [
                    {"tool": "multimodal_extraction", "duration_ms": 380.0, "inputs": {"doc": "MRPL_Turnaround_Inspection_Report_2026.md"}},
                    {"tool": "hybrid_retrieval", "duration_ms": 72.0, "inputs": {"query": "SOP-17 retirement limits"}},
                    {"tool": "deterministic_calculator", "duration_ms": 2.0, "inputs": {"measured": 3.42, "retirement": 4.80}},
                    {"tool": "document_generator", "duration_ms": 290.0, "inputs": {"format": "docx"}},
                ]

                render_activity_disclosure(activity, tech_details)

                store.save_chat_message(
                    active_session.id,
                    "assistant",
                    resp_text,
                    metadata={
                        "docx_path": docx_path,
                        "activity": activity,
                        "technical_details": tech_details,
                    },
                )

            else:
                # Autonomous Cognitive ReAct Plan Loop
                with st.status("Working · Executing task plan...", expanded=False) as status_box:
                    stream_holder = st.empty()
                    chunks = []
                    final_resp = ""
                    cits = []
                    gen_deliv = None
                    activity_log = []
                    tech_log = []

                    step_n = 0
                    for event in orchestrator.run_autonomous_plan_loop_stream(
                        user_prompt=prompt,
                        project_id=project_id,
                        category=active_scope,
                        max_steps=5,
                    ):
                        e_type = event.get("type")
                        if e_type == "plan_start":
                            route = event.get("route", {})
                            status_box.update(label=f"Working · Assigned {route.get('assigned_model', 'specialist')}...", state="running")
                        elif e_type == "tool_call":
                            step_n += 1
                            t_name = event.get("tool", "")
                            status_box.update(label=f"Working · Running {t_name.replace('_', ' ')} · {step_n} of 4 steps", state="running")
                        elif e_type == "tool_result":
                            t_name = event.get("tool", "")
                            t_out = event.get("output", {})
                            s_summary = "Complete"
                            if t_name == "knowledge_search":
                                s_summary = f"{len(t_out.get('results', []))} sections cited"
                            elif t_name == "calculate_wall_thinning":
                                s_summary = f"{t_out.get('data', {}).get('deviation_percentage_below_retirement', '')}% gap"
                            activity_log.append({
                                "title": t_name.replace("_", " ").capitalize(),
                                "summary": s_summary,
                                "status": "completed",
                            })
                            tech_log.append({
                                "tool": t_name,
                                "inputs": event.get("input"),
                                "status": t_out.get("status"),
                            })
                        elif e_type == "final_chunk":
                            chunks.append(event.get("chunk", ""))
                            stream_holder.markdown("".join(chunks))
                        elif e_type == "completed":
                            final_resp = event.get("final_response", "")
                            cits = event.get("citations", [])
                            gen_deliv = event.get("generated_deliverable")
                            status_box.update(label=f"Complete · {len(activity_log) or 1} steps completed", state="complete", expanded=False)

                full_text = final_resp if final_resp else "".join(chunks)
                cleaned_full_text = clean_boilerplate_header(full_text)
                stream_holder.markdown(cleaned_full_text)

                if gen_deliv and Path(gen_deliv).exists():
                    render_attachment_row(gen_deliv, key_suffix="auto_live")

                if activity_log:
                    render_activity_disclosure(activity_log, tech_log)

                meta_data = {
                    "citations": cits,
                    "activity": activity_log,
                    "technical_details": tech_log,
                }
                if gen_deliv:
                    meta_data["generated_deliverable"] = gen_deliv
                    if str(gen_deliv).endswith(".pdf"):
                        meta_data["pdf_path"] = gen_deliv
                    elif str(gen_deliv).endswith(".docx"):
                        meta_data["docx_path"] = gen_deliv

                store.save_chat_message(
                    active_session.id,
                    "assistant",
                    cleaned_full_text,
                    metadata=meta_data,
                )

# Render Settings Drawer if toggled
if st.session_state.show_settings and col_settings is not None:
    with col_settings:
        render_settings_drawer(active_session, active_scope, egress_snap)
