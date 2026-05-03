import os
import json
import streamlit as st
import anthropic
import requests
from bs4 import BeautifulSoup

MODEL = "claude-haiku-4-5-20251001"  # analysis (high rate limit, no tool use needed)
SEARCH_MODEL = "claude-sonnet-4-6"   # civic resources web search (requires tool use)
MAX_TEXT_CHARS = 5000

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "article_topic": {"type": "string"},
        "key_terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "definition": {"type": "string"}
                },
                "required": ["term", "definition"],
                "additionalProperties": False
            }
        },
        "annotated_segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": ["factual", "opinion", "neutral"]}
                },
                "required": ["text", "type"],
                "additionalProperties": False
            }
        },
        "factual_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "why_factual": {"type": "string"}
                },
                "required": ["claim", "why_factual"],
                "additionalProperties": False
            }
        },
        "opinion_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "why_opinion": {"type": "string"}
                },
                "required": ["claim", "why_opinion"],
                "additionalProperties": False
            }
        }
    },
    "required": ["article_topic", "key_terms", "annotated_segments", "factual_claims", "opinion_claims"],
    "additionalProperties": False
}


def fetch_article_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsLens/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    container = soup.find("article") or soup.find("main") or soup.body
    if container is None:
        return ""
    paragraphs = container.find_all("p")
    return "\n".join(p.get_text(" ", strip=True) for p in paragraphs)


def analyze_article(client: anthropic.Anthropic, text: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        output_config={"format": {"type": "json_schema", "schema": ANALYSIS_SCHEMA}},
        messages=[{
            "role": "user",
            "content": f"""Analyze this news article and return a structured JSON response.

1. GLOSSARY (key_terms): Identify 3–6 technical, legal, political, or domain-specific terms
   that an average reader might not know. For each, provide a plain-English definition of
   1–2 sentences.

2. ANNOTATED SUMMARY (annotated_segments): Write a flowing 8–12 sentence summary of the
   article. Break it into sentence-level segments. Tag each segment as:
   - "factual": verifiable statements — specific events with dates, official statistics,
     direct quotes with attribution, documented actions by named entities
   - "opinion": interpretations, characterizations, value judgments, predictions,
     or claims reasonable people could disagree about
   - "neutral": background context, transitions, or descriptive statements that are
     neither clearly factual nor opinion

   The segments must read naturally when concatenated. Do NOT add labels in the text itself.

3. CLAIMS LISTS (factual_claims, opinion_claims): Also extract 3–6 standalone factual
   claims and 3–6 standalone opinion claims with brief explanations of your classification.

Article:
{text[:MAX_TEXT_CHARS]}"""
        }]
    )
    if response.stop_reason == "refusal":
        raise ValueError("Unable to analyze this content.")
    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        raise ValueError("No analysis returned.")
    return json.loads(text_block.text)


def _run_with_web_search(client: anthropic.Anthropic, messages: list) -> str:
    """Run a web-search-enabled conversation and return the final text response."""
    response = None
    for _ in range(5):  # max 5 continue-after-pause_turn rounds
        response = client.messages.create(
            model=SEARCH_MODEL,
            max_tokens=4096,
            tools=[{"type": "web_search_20260209", "name": "web_search"}],
            messages=messages
        )
        if response.stop_reason == "end_turn":
            break
        # pause_turn means the server-side search loop hit its limit — re-send to continue
        messages.append({"role": "assistant", "content": response.content})

    if response is None:
        return ""
    return next((b.text for b in response.content if b.type == "text"), "")



def get_civic_resources(
    client: anthropic.Anthropic,
    user_opinion: str,
    topic: str,
    opinion_claims: list
) -> str:
    claims_text = "\n".join(f"- {c['claim']}" for c in opinion_claims[:3])
    messages = [{
        "role": "user",
        "content": f"""Someone has read a news article about: {topic}

The article raised these opinion-based issues:
{claims_text}

Their personal stance: {user_opinion}

Search the web for concrete, actionable resources to help them get involved:
1. Advocacy organizations they can join or support
2. Active petitions or campaigns to sign
3. How to contact their elected representatives (link to a rep-finder tool)
4. Upcoming events, town halls, or rallies
5. Ways to volunteer or donate

Prioritize real, current resources with working URLs."""
    }]
    return _run_with_web_search(client, messages)


def render_annotated_summary(segments: list) -> str:
    COLORS = {
        "factual": "#d4edda",   # green
        "opinion": "#f8d7da",   # red/pink
        "neutral": "transparent",
    }
    parts = []
    for seg in segments:
        color = COLORS.get(seg["type"], "transparent")
        text = seg["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if color == "transparent":
            parts.append(f'<span style="line-height:1.8">{text}</span>')
        else:
            parts.append(
                f'<span style="background-color:{color};padding:2px 4px;'
                f'border-radius:3px;line-height:1.8">{text}</span>'
            )
    return '<p style="font-size:1.05rem;line-height:2">' + " ".join(parts) + "</p>"


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="NewsLens", page_icon="🔍", layout="wide")

st.title("🔍 NewsLens")
st.markdown("*Read smarter. Think critically. Act meaningfully.*")

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error(
        "Please set your `ANTHROPIC_API_KEY` environment variable and restart the app.\n\n"
        "```bash\nexport ANTHROPIC_API_KEY=your-key-here\nstreamlit run app.py\n```"
    )
    st.stop()


@st.cache_resource
def get_client():
    return anthropic.Anthropic()


client = get_client()

for key in ["analysis", "resources", "current_url"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Step 1: URL Input ──────────────────────────────────────────────────────────
st.header("1 · Enter a News Article URL")
st.caption("Paste the link to any publicly accessible news article")

url_input = st.text_input("Article URL", placeholder="https://www.example.com/news/article")

if url_input:
    if url_input != st.session_state.current_url:
        st.session_state.current_url = url_input
        st.session_state.analysis = None
        st.session_state.research = None
        st.session_state.resources = None

    if st.session_state.analysis is None:
        if st.button("Analyze Article →", type="primary"):
            with st.spinner("Fetching article and identifying claims…"):
                try:
                    text = fetch_article_text(url_input)
                    if not text.strip():
                        st.error("Could not extract text from this page. The site may block scrapers or require a login.")
                        st.stop()
                    st.session_state.analysis = analyze_article(client, text)
                except requests.RequestException as e:
                    st.error(f"Could not fetch the article: {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.stop()
            st.rerun()

# ── Step 2: Analysis ──────────────────────────────────────────────────────────
if st.session_state.analysis:
    a = st.session_state.analysis

    st.divider()
    st.header("2 · Article Analysis")
    st.caption(f"**Topic:** {a['article_topic']}")

    # Key terms glossary
    if a.get("key_terms"):
        with st.expander("📖 Key Terms Glossary", expanded=True):
            for kt in a["key_terms"]:
                st.markdown(f"**{kt['term']}** — {kt['definition']}")

    # Color-coded annotated summary
    st.subheader("Annotated Summary")
    st.markdown(
        '<span style="background-color:#d4edda;padding:2px 6px;border-radius:3px">■ Factual</span>'
        '&nbsp;&nbsp;'
        '<span style="background-color:#f8d7da;padding:2px 6px;border-radius:3px">■ Opinion-based</span>'
        '&nbsp;&nbsp;Uncolored = neutral context',
        unsafe_allow_html=True
    )
    st.markdown(render_annotated_summary(a["annotated_segments"]), unsafe_allow_html=True)

    # Expandable claim details
    with st.expander("✅ Factual Claims — details"):
        for item in a["factual_claims"]:
            st.markdown(f"- **{item['claim']}**  \n  *{item['why_factual']}*")

    with st.expander("💬 Opinion-Based Claims — details"):
        for item in a["opinion_claims"]:
            st.markdown(f"- **{item['claim']}**  \n  *{item['why_opinion']}*")

    # ── Step 3: Your Opinion & How to Act ─────────────────────────────────────
    st.divider()
    st.header("3 · Your Opinion & How to Act")

    user_opinion = st.text_area(
        "What's your take on the issues raised in this article?",
        placeholder="e.g., 'I think renewable energy subsidies should prioritize rural communities…'",
        height=100,
        key="opinion_input"
    )

    if user_opinion:
        if st.button("Find Ways to Get Involved →", type="primary"):
            with st.spinner("Searching for civic engagement opportunities…"):
                try:
                    st.session_state.resources = get_civic_resources(
                        client,
                        user_opinion,
                        a["article_topic"],
                        a["opinion_claims"]
                    )
                except Exception as e:
                    st.error(f"Resource search failed: {e}")
                    st.stop()
            st.rerun()

    # ── Step 4: Civic Resources ────────────────────────────────────────────────
    if st.session_state.resources:
        st.divider()
        st.header("4 · Ways to Get Involved")
        st.markdown(st.session_state.resources)
        st.success(
            "Ready to make your voice heard? "
            "Use the resources above to connect with people working on these issues."
        )
