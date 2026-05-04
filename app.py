import os
import json
import streamlit as st
import anthropic
import requests
from bs4 import BeautifulSoup

MODEL = "claude-haiku-4-5-20251001"
SEARCH_MODEL = "claude-sonnet-4-6"
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
        },
    },
    "required": ["article_topic", "key_terms", "annotated_segments", "factual_claims", "opinion_claims"],
    "additionalProperties": False
}


RESOURCES_SCHEMA = {
    "type": "object",
    "properties": {
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "url": {"type": "string"},
                    "time_score": {"type": "integer"},
                    "impact_score": {"type": "integer"}
                },
                "required": ["title", "description", "url", "time_score", "impact_score"],
                "additionalProperties": False
            }
        }
    },
    "required": ["options"],
    "additionalProperties": False
}

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "feedback": {"type": "string"}
    },
    "required": ["score", "feedback"],
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


def find_counter_articles(
    client: anthropic.Anthropic,
    opinion_claims: list,
    topic: str
) -> str:
    claims_text = "\n".join(f"- {c['claim']}" for c in opinion_claims[:2])
    response = client.messages.create(
        model=SEARCH_MODEL,
        max_tokens=1024,
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
        messages=[{
            "role": "user",
            "content": f"""Search for 2 real news articles that counter these claims about {topic}:
{claims_text}

For each: title as a markdown link, publication + date, one short quote, which claim it counters."""
        }]
    )
    return next((b.text for b in response.content if b.type == "text"), "")


def score_opinion(
    client: anthropic.Anthropic,
    user_opinion: str,
    topic: str,
    opinion_claims: list
) -> dict:
    claims_text = "\n".join(f"- {c['claim']}" for c in opinion_claims[:4])
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        output_config={"format": {"type": "json_schema", "schema": SCORE_SCHEMA}},
        messages=[{
            "role": "user",
            "content": f"""Score this opinion on nuance and consideration of multiple perspectives.

Topic: {topic}

Opinion-based claims in the article (representing different angles on this issue):
{claims_text}

User's opinion: "{user_opinion}"

Score 0–10 using this rubric:
- 0–2: Entirely one-sided, no acknowledgment of complexity or opposing views
- 3–4: Slight awareness of other views but mostly one-sided
- 5–6: Acknowledges multiple perspectives, even if favoring one side
- 7–8: Thoughtful, engages with trade-offs or counterarguments
- 9–10: Highly nuanced, considers limitations of own position

Return:
- score: integer 0–10
- feedback: 1–2 sentences explaining the score and concretely what would raise it"""
        }]
    )
    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        raise ValueError("No score returned.")
    result = json.loads(text_block.text)
    result["score"] = max(0, min(10, int(result["score"])))
    return result


def get_civic_resources(
    client: anthropic.Anthropic,
    user_opinion: str,
    topic: str,
    opinion_claims: list
) -> list:
    claims_text = "\n".join(f"- {c['claim']}" for c in opinion_claims[:3])
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        output_config={"format": {"type": "json_schema", "schema": RESOURCES_SCHEMA}},
        messages=[{
            "role": "user",
            "content": f"""Someone has read a news article about: {topic}

The article raised these opinion-based issues:
{claims_text}

Their personal stance: {user_opinion}

Suggest exactly 3 concrete, actionable civic engagement options. For each option provide:
- title: short name of the action or organization
- description: 1–2 sentences on what it is and how to take action
- url: a real, working website URL
- time_score: 0–10 for how much time it requires (0 = minutes, 10 = ongoing major commitment)
- impact_score: 0–10 for potential civic impact (0 = minimal, 10 = highly impactful)

Use well-known, reliable organizations and government resources (e.g., vote.gov, congress.gov)."""
        }]
    )
    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        raise ValueError("No resources returned.")
    data = json.loads(text_block.text)
    options = data.get("options", [])[:3]
    for opt in options:
        opt["time_score"] = max(0, min(10, int(opt["time_score"])))
        opt["impact_score"] = max(0, min(10, int(opt["impact_score"])))
    return options


def chat_about_article(
    client: anthropic.Anthropic,
    history: list,
    article_text: str,
    analysis: dict
) -> str:
    opinion_claims = "\n".join(f"- {c['claim']}" for c in analysis["opinion_claims"][:4])
    system = f"""You are a balanced, non-judgmental guide helping a user understand a news article and the broader context around it. You can search the web to find additional information, statistics, expert opinions, or recent developments.

Article topic: {analysis["article_topic"]}

Article text:
{article_text[:3000]}

Opinion-based claims in the article:
{opinion_claims}

Help the user explore all sides of the issues fairly. Search the web when the question benefits from current information or sources beyond the article. Never push your own opinion."""
    messages = list(history)
    response = None
    for _ in range(5):
        response = client.messages.create(
            model=SEARCH_MODEL,
            max_tokens=1024,
            system=system,
            tools=[{"type": "web_search_20260209", "name": "web_search"}],
            messages=messages
        )
        if response.stop_reason == "end_turn":
            break
        messages.append({"role": "assistant", "content": response.content})
    if response is None:
        return ""
    return next((b.text for b in response.content if b.type == "text"), "")


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

for key in ["analysis", "resources", "article_text", "current_url", "opinion_score", "opinion_feedback", "counter_sources"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Step 1: URL Input ──────────────────────────────────────────────────────────
st.header("1 · Enter a News Article URL")
st.caption("Paste the link to any publicly accessible news article")

url_input = st.text_input("Article URL", placeholder="https://www.example.com/news/article")

if url_input:
    if url_input != st.session_state.current_url:
        st.session_state.current_url = url_input
        st.session_state.analysis = None
        st.session_state.resources = None
        st.session_state.article_text = None
        st.session_state.chat_history = []
        st.session_state.opinion_score = None
        st.session_state.opinion_feedback = None
        st.session_state.counter_sources = None

    if st.session_state.analysis is None:
        if st.button("Analyze Article →", type="primary"):
            with st.spinner("Fetching article and identifying claims…"):
                try:
                    text = fetch_article_text(url_input)
                    if not text.strip():
                        st.error("Could not extract text from this page. The site may block scrapers or require a login.")
                        st.stop()
                    st.session_state.article_text = text
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

    # Counter-opinion sources
    st.subheader("Other Perspectives")
    if st.session_state.counter_sources:
        st.caption("Real articles presenting views that counter the opinion-based claims in this piece")
        st.markdown(st.session_state.counter_sources)
    else:
        if st.button("Find Counter Articles →", type="secondary"):
            with st.spinner("Searching for counter sources… (this may take ~20 seconds)"):
                try:
                    st.session_state.counter_sources = find_counter_articles(
                        client, a["opinion_claims"], a["article_topic"]
                    )
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.stop()
            st.rerun()

    # ── Step 3: Q&A Chatbot ────────────────────────────────────────────────────
    st.divider()
    st.header("3 · Ask Questions About This Article")
    st.caption("Explore different perspectives — ask anything about the article, the claims, or the issues it raises.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        user_question = st.text_input("", placeholder="Ask a question about this article…",
                                      key="chat_q", label_visibility="collapsed")
    with col_btn:
        ask_clicked = st.button("Ask →", use_container_width=True)

    if ask_clicked and user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.spinner("Thinking…"):
            reply = chat_about_article(
                client,
                st.session_state.chat_history,
                st.session_state.article_text or "",
                a
            )
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    # ── Step 4: Your Opinion & How to Act ─────────────────────────────────────
    st.divider()
    st.header("4 · Your Opinion & How to Act")
    st.caption("Share your take — we'll score it for nuance before connecting you with resources.")

    user_opinion = st.text_area(
        "What's your take on the issues raised in this article?",
        placeholder="e.g., 'I think renewable energy subsidies should prioritize rural communities, though I recognize the economic concerns for urban areas…'",
        height=120,
        key="opinion_input"
    )

    if user_opinion:
        if st.button("Score My Opinion →", type="primary"):
            with st.spinner("Evaluating nuance…"):
                try:
                    result = score_opinion(client, user_opinion, a["article_topic"], a["opinion_claims"])
                    st.session_state.opinion_score = result["score"]
                    st.session_state.opinion_feedback = result["feedback"]
                    st.session_state.resources = None
                except Exception as e:
                    st.error(f"Scoring failed: {e}")
                    st.stop()
            st.rerun()

    if st.session_state.opinion_score is not None:
        score = st.session_state.opinion_score
        col_score, col_bar = st.columns([1, 4])
        with col_score:
            st.metric("Nuance Score", f"{score} / 10")
        with col_bar:
            st.progress(score / 10)

        if score < 5:
            st.warning(
                f"**{st.session_state.opinion_feedback}**\n\n"
                "Revise your opinion above and re-score to reach 5/10 to unlock civic resources."
            )
        else:
            st.success(f"{st.session_state.opinion_feedback}")
            if st.session_state.resources is None:
                if st.button("Find Ways to Get Involved →", type="primary"):
                    with st.spinner("Finding civic engagement resources…"):
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

    # ── Step 5: Ways to Get Involved ──────────────────────────────────────────
    if st.session_state.resources:
        st.divider()
        st.header("5 · Ways to Get Involved")
        st.caption("⏱ Time = how much time it takes (0 = minutes, 10 = major commitment)  ·  💥 Impact = potential civic impact (0 = low, 10 = high)")
        for opt in st.session_state.resources:
            with st.container(border=True):
                st.markdown(f"### [{opt['title']}]({opt['url']})")
                st.markdown(opt["description"])
                col_t, col_i = st.columns(2)
                with col_t:
                    st.caption("⏱ Time required")
                    st.progress(opt["time_score"] / 10)
                    st.markdown(f"**{opt['time_score']} / 10**")
                with col_i:
                    st.caption("💥 Impact")
                    st.progress(opt["impact_score"] / 10)
                    st.markdown(f"**{opt['impact_score']} / 10**")
