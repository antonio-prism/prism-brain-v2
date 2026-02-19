"""
PRISM Brain - Shared Theme
===========================
Visual design aligned with the PRISM corporate website.
Import inject_prism_theme() at the top of each page for consistent look.

Design tokens extracted from https://prism website:
  Primary dark teal:  #1B3A4B
  Accent orange:      #E8862A
  Light background:   #F5F7FA
  Card white:         #FFFFFF
  Body text:          #3D4F5F
  Muted text:         #7A8D9C
"""

import streamlit as st


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLORS = {
    # Backgrounds
    "bg": "#F5F7FA",
    "card": "#FFFFFF",
    "border": "#E0E6ED",

    # Primary palette
    "primary": "#1B3A4B",       # dark teal (headings, sidebar, header)
    "primary_light": "#24506A", # lighter teal for hover
    "accent": "#E8862A",        # orange (CTAs, highlights, accents)
    "accent_hover": "#D17622",  # darker orange on hover

    # Text
    "text": "#1B3A4B",
    "text_body": "#3D4F5F",
    "muted": "#7A8D9C",
    "light": "#A0B1BF",

    # Semantic
    "green": "#22c55e",
    "red": "#ef4444",
    "yellow": "#eab308",
    "blue": "#3b82f6",
    "purple": "#8b5cf6",
    "teal": "#14b8a6",
}

DOMAIN_COLORS = {
    "PHYSICAL": "#E8862A",     # orange  – matches PRISM accent
    "STRUCTURAL": "#1B3A4B",   # dark teal – matches PRISM primary
    "DIGITAL": "#7030A0",      # purple
    "OPERATIONAL": "#22876C",  # green-teal
}

DOMAIN_ICONS = {
    "PHYSICAL": "🌍",
    "STRUCTURAL": "🏛️",
    "DIGITAL": "💻",
    "OPERATIONAL": "⚙️",
}

RISK_BADGE = {
    "HIGH":     {"bg": "#fef2f2", "fg": "#dc2626"},
    "ELEVATED": {"bg": "#fff7ed", "fg": "#ea580c"},
    "MODERATE": {"bg": "#fefce8", "fg": "#ca8a04"},
    "LOW":      {"bg": "#f0fdf4", "fg": "#16a34a"},
}


# ---------------------------------------------------------------------------
# Core CSS injection
# ---------------------------------------------------------------------------
_LOGO_B64 = (
    "PHN2ZyB3aWR0aD0iMTgyIiBoZWlnaHQ9IjIzMyIgdmlld0JveD0iMCAwIDE4MiAyMzMiIGZpbGw9"
    "Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxnIGNsaXAtcGF0aD0i"
    "dXJsKCNjbGlwMF8xMjFfMzQ0KSI+CjxwYXRoIGQ9Ik0wIDBWMjMzSDE4MlYwSDBaTTkuMTczMDcg"
    "OS4wNDAxNUgxNzIuODM0VjEzNi4yNjJMOTAuMzYzNCA2Mi44ODU1TDg0LjcyMDYgNzMuMDY4NEw5"
    "LjE3MzA3IDIwNy41NVY5LjA0MDE1Wk0xNzIuODM0IDE1MC42NVYyMDkuNjZMMTA4LjExNiA5NC4x"
    "MzA0TDE3Mi44MzQgMTUwLjY1Wk0xNC4xOTM3IDIyMy45Nkw5MC4zMjcyIDg3LjAwMjNMMTY2Ljgw"
    "OCAyMjMuOTk2TDE0LjE5MzcgMjIzLjk2WiIgZmlsbD0id2hpdGUiLz4KPC9nPgo8ZGVmcz4KPGNs"
    "aXBQYXRoIGlkPSJjbGlwMF8xMjFfMzQ0Ij4KPHJlY3Qgd2lkdGg9IjE4MiIgaGVpZ2h0PSIyMzMi"
    "IGZpbGw9IndoaXRlIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg=="
)


def inject_prism_theme():
    """Inject PRISM theme CSS (including sidebar logo via CSS) into the current page."""
    st.markdown(_PRISM_CSS, unsafe_allow_html=True)


_PRISM_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════════════
   PRISM Brain – Corporate Theme
   Aligned with the PRISM website design language.
   Primary: #1B3A4B (dark teal)  ·  Accent: #E8862A (orange)
   ═══════════════════════════════════════════════════════════════════════ */

/* ── Fonts & base ──────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Roboto, 'Helvetica Neue', Arial, sans-serif;
    color: #3D4F5F;
}

/* ── Main background ──────────────────────────────────────────────── */
.stApp {
    background-color: #F5F7FA;
}

/* ── Top header bar ───────────────────────────────────────────────── */
header[data-testid="stHeader"] {
    background: #1B3A4B !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B3A4B 0%, #152E3D 100%);
    border-right: 1px solid #0F2633;
}
/* Logo + app name pinned above navigation */
section[data-testid="stSidebar"]::before {
    content: "PRISM Brain";
    display: block;
    background-image: url("data:image/svg+xml;base64,""" + _LOGO_B64 + """");
    background-repeat: no-repeat;
    background-position: 24px center;
    background-size: 28px auto;
    padding: 18px 24px 18px 62px;
    margin: 0 0 8px 0;
    border-bottom: 1px solid #2A5266;
    color: #FFFFFF !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    line-height: 28px;
}
section[data-testid="stSidebar"] * {
    color: #B8CCDA !important;
}
section[data-testid="stSidebar"] .stMarkdown strong,
section[data-testid="stSidebar"] .stMarkdown b,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #2A5266 !important;
}
section[data-testid="stSidebar"] .stCaption {
    color: #7A9AAE !important;
}
/* Sidebar selectbox / inputs */
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label {
    color: #B8CCDA !important;
}

/* ── Headings ─────────────────────────────────────────────────────── */
h1, h2, h3 {
    color: #1B3A4B !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}
h4, h5, h6 {
    color: #24506A !important;
    font-weight: 600 !important;
}

/* ── Metric cards ─────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E0E6ED;
    border-left: 4px solid #E8862A;
    border-radius: 10px;
    padding: 18px 22px;
    box-shadow: 0 1px 4px rgba(27,58,75,0.06);
}
div[data-testid="stMetric"] label {
    color: #7A8D9C !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #1B3A4B !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}

/* ── Primary buttons (orange accent) ──────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: #E8862A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.4rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 6px rgba(232,134,42,0.25);
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #D17622 !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(232,134,42,0.35);
    transform: translateY(-1px);
}

/* ── Secondary / default buttons (teal outline) ──────────────────── */
.stButton > button {
    background: transparent;
    color: #1B3A4B;
    border: 1px solid #1B3A4B;
    border-radius: 8px;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #1B3A4B;
    color: #FFFFFF;
    border-color: #1B3A4B;
}

/* ── Form inputs ──────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea textarea {
    border-radius: 8px !important;
    border-color: #E0E6ED !important;
    color: #3D4F5F !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #E8862A !important;
    box-shadow: 0 0 0 2px rgba(232,134,42,0.15) !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    border-bottom: 2px solid #E0E6ED;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    color: #7A8D9C;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    color: #1B3A4B !important;
    font-weight: 600;
    border-bottom-color: #E8862A !important;
}

/* ── Expanders ────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #1B3A4B;
    background: #FFFFFF;
    border-radius: 8px;
}
details[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E0E6ED;
    border-radius: 10px;
    margin-bottom: 8px;
}

/* ── Dataframe / table ────────────────────────────────────────────── */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #E0E6ED;
}

/* ── Divider ──────────────────────────────────────────────────────── */
hr {
    border-color: #E0E6ED !important;
}

/* ── Alert boxes ──────────────────────────────────────────────────── */
.stAlert {
    border-radius: 8px;
}

/* ── Links ────────────────────────────────────────────────────────── */
a { color: #E8862A; }
a:hover { color: #D17622; }

/* ── Download button ──────────────────────────────────────────────── */
.stDownloadButton > button {
    background: #1B3A4B !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px;
    font-weight: 500;
}
.stDownloadButton > button:hover {
    background: #24506A !important;
    color: #FFFFFF !important;
}

/* ── Checkboxes – orange accent ───────────────────────────────────── */
.stCheckbox [data-testid="stCheckbox"] span[role="checkbox"][aria-checked="true"] {
    background-color: #E8862A !important;
    border-color: #E8862A !important;
}

/* ── Progress bar ─────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background-color: #E8862A !important;
}

/* ═══════════════════════════════════════════════════════════════════════
   Helper classes for custom HTML components
   ═══════════════════════════════════════════════════════════════════════ */

/* Page title */
.prism-page-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1B3A4B;
    letter-spacing: -0.3px;
    margin-bottom: 4px;
}
.prism-page-subtitle {
    font-size: 0.95rem;
    color: #7A8D9C;
    margin-bottom: 1rem;
}

/* Domain card with left accent border */
.prism-domain-card {
    background: #FFFFFF;
    border: 1px solid #E0E6ED;
    border-radius: 10px;
    padding: 20px;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.prism-domain-card:hover {
    box-shadow: 0 4px 16px rgba(27,58,75,0.10);
    transform: translateY(-2px);
}

/* Stat highlight card (used on welcome page) */
.prism-stat-card {
    border-radius: 10px;
    padding: 22px;
    color: white;
}
.prism-stat-card .label {
    font-size: 12px;
    opacity: 0.85;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
}
.prism-stat-card .value {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 6px 0;
}
.prism-stat-card .detail {
    font-size: 12px;
    opacity: 0.7;
}

/* Risk badge */
.prism-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
}

/* Footer */
.prism-footer {
    text-align: center;
    padding: 24px 0;
    color: #A0B1BF;
    font-size: 12px;
    margin-top: 48px;
    border-top: 1px solid #E0E6ED;
}

/* Info cards with left orange border (like PRISM website services) */
.prism-info-card {
    background: #FFFFFF;
    border: 1px solid #E0E6ED;
    border-left: 4px solid #E8862A;
    border-radius: 0 10px 10px 0;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.prism-info-card h4 {
    color: #1B3A4B !important;
    margin: 0 0 8px 0;
}
.prism-info-card p {
    color: #3D4F5F;
    margin: 0;
    font-size: 0.92rem;
}

</style>
"""


# ---------------------------------------------------------------------------
# Page-header helper
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str = "", icon: str = ""):
    """Render a consistent PRISM page header."""
    prefix = f"{icon} " if icon else ""
    st.markdown(
        f'<p class="prism-page-title">{prefix}{title}</p>'
        f'<p class="prism-page-subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def page_footer(version: str = "2.0.0"):
    """Render a consistent PRISM footer."""
    st.markdown(
        f'<div class="prism-footer">'
        f'PRISM Brain v{version} · © 2026 · Risk Intelligence System'
        f'</div>',
        unsafe_allow_html=True,
    )


def domain_card_html(domain: str, event_count: int, family_count: int, extra: str = ""):
    """Return styled HTML for a domain card with left accent border."""
    color = DOMAIN_COLORS.get(domain, "#888")
    icon = DOMAIN_ICONS.get(domain, "📊")
    label = domain.capitalize()
    return f"""
    <div style="background:#FFFFFF; border:1px solid #E0E6ED;
                border-left:4px solid {color};
                border-radius:0 10px 10px 0; padding:18px 22px;
                min-height:100px; transition: box-shadow 0.2s ease;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
            <span style="font-size:1.5rem;">{icon}</span>
            <strong style="font-size:1.05rem; color:{color};">{label}</strong>
        </div>
        <span style="font-size:0.88rem; color:#7A8D9C;">
            {event_count} events · {family_count} families
        </span>
        {f'<br><span style="font-size:0.82rem; color:#A0B1BF;">{extra}</span>' if extra else ''}
    </div>
    """
