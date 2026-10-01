"""SentiScope - a neural sentiment analyser that explains itself, word by word."""
import html
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import ART, MAXLEN, encode, tokenize

try:
    from st_keyup import st_keyup
except Exception:  # component missing -> fall back to paste mode only
    st_keyup = None

POS, NEG, AMB = "#2DD4BF", "#FB7185", "#FBBF24"
TEXT, MUTED, CARD, LINE = "#E6EAF2", "#8B97AD", "#111A2E", "#1E2A44"
OCCLUSION_CAP = 120

EXAMPLES = {
    "Glowing review": "An absolutely brilliant film. The performances are heartfelt, the story is gripping and the music is beautiful.",
    "Scathing review": "A tedious, poorly written mess. The acting is wooden, the plot makes no sense and I wanted my money back.",
    "Mixed feelings": "The visuals are stunning but the story is dull and the ending was a disappointment.",
    "Plot twist": "I expected a terrible movie, but it was surprisingly good and I loved every minute.",
}

st.set_page_config(page_title="SentiScope", page_icon="◐", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}
.hero h1 {{ font-size: 2.4rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; color: {TEXT}; }}
.hero p {{ color: {MUTED}; font-size: 1.05rem; margin: .35rem 0 1.4rem; }}
.card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 1.2rem 1.4rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: {CARD}; border: 1px solid {LINE} !important; border-radius: 16px; }}
.ptitle {{ margin: 0 0 .4rem; font-size: .8rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: {MUTED}; }}
.card h4 {{ margin: 0 0 .6rem; font-size: .8rem; font-weight: 600; letter-spacing: .08em;
           text-transform: uppercase; color: {MUTED}; }}
.metric {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 1rem 1.2rem; }}
.metric .v {{ font-size: 1.8rem; font-weight: 700; color: {TEXT}; }}
.metric .l {{ font-size: .78rem; color: {MUTED}; text-transform: uppercase; letter-spacing: .08em; }}
.verdict {{ display: inline-block; padding: .35rem .9rem; border-radius: 999px; font-weight: 600; font-size: .95rem; }}
.hl {{ line-height: 2.3; font-size: 1.12rem; }}
.hl span {{ padding: 3px 6px; border-radius: 7px; margin: 0 1px; }}
.legend {{ color: {MUTED}; font-size: .82rem; margin-top: .8rem; }}
.step {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: 1rem 1.1rem; height: 100%; }}
.step b {{ color: {POS}; font-size: .8rem; letter-spacing: .08em; }}
.step div {{ font-weight: 600; margin: .2rem 0 .3rem; }}
.step small {{ color: {MUTED}; line-height: 1.5; display: block; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: .6rem 1.1rem; border-radius: 10px 10px 0 0; color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {TEXT} !important; }}
div.stButton > button {{ background: {CARD}; border: 1px solid {LINE}; color: {TEXT}; border-radius: 10px; }}
div.stButton > button:hover {{ border-color: {POS}; color: {POS}; }}
.stTextArea textarea, .stTextInput input {{ background: {CARD} !important; border-radius: 12px !important; }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading neural network...")
def load_assets():
    from tensorflow import keras

    if not (ART / "model.keras").exists():
        import train

        train.main()
    model = keras.models.load_model(ART / "model.keras", compile=False)
    vocab = json.loads((ART / "vocab.json").read_text())
    metrics = json.loads((ART / "metrics.json").read_text())
    return model, vocab, metrics


def predict(model, vocab, token_lists):
    from tensorflow import keras

    seqs = [encode(w, vocab) for w in token_lists]
    x = keras.utils.pad_sequences(seqs, maxlen=MAXLEN)
    return np.asarray(model(x, training=False)).ravel()


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def analyse(text, model, vocab):
    words = tokenize(text)[-MAXLEN:]
    if not words:
        return None
    cap = min(len(words), OCCLUSION_CAP)
    variants = [words[:i] + words[i + 1 :] for i in range(cap)]
    probs = predict(model, vocab, [words] + variants)
    p = float(probs[0])
    impact = np.zeros(len(words))
    impact[:cap] = logit(probs[0]) - logit(probs[1:])  # >0 pushes toward positive
    peak = float(np.max(np.abs(impact))) or 1.0
    return {"words": words, "p": p, "impact": impact, "norm": impact / peak}


# ---------------------------------------------------------------- visuals
def verdict(p):
    if p >= 0.6:
        return "Positive", POS
    if p <= 0.4:
        return "Negative", NEG
    return "Mixed", AMB


def gauge(p):
    _, color = verdict(p)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(p * 100, 1),
            number={"suffix": "%", "font": {"size": 46, "color": TEXT}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(251,113,133,.14)"},
                    {"range": [40, 60], "color": "rgba(251,191,36,.12)"},
                    {"range": [60, 100], "color": "rgba(45,212,191,.14)"},
                ],
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=24, r=24, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def highlighted(res):
    parts = []
    for w, n, raw in zip(res["words"], res["norm"], res["impact"]):
        if abs(n) < 0.08:
            parts.append(f"<span>{html.escape(w)}</span>")
            continue
        rgb = "45,212,191" if n > 0 else "251,113,133"
        alpha = min(0.18 + 0.62 * abs(n), 0.8)
        parts.append(
            f'<span style="background:rgba({rgb},{alpha:.2f})" title="{raw:+.2f}">{html.escape(w)}</span>'
        )
    return '<div class="hl">' + " ".join(parts) + "</div>"


def influence_chart(res, top=8):
    df = pd.DataFrame({"word": res["words"], "impact": res["impact"]})
    df = df.groupby("word", as_index=False)["impact"].sum()
    df = df.reindex(df["impact"].abs().sort_values(ascending=False).index).head(top).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=df["impact"], y=df["word"], orientation="h",
            marker_color=[POS if v > 0 else NEG for v in df["impact"]],
        )
    )
    fig.update_layout(
        height=max(220, 34 * len(df) + 60), margin=dict(l=0, r=10, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter"),
        xaxis=dict(gridcolor=LINE, zerolinecolor=MUTED, title="Influence on prediction"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def card(title, body):
    st.markdown(f'<div class="card"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


def metric(col, value, label):
    col.markdown(f'<div class="metric"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


def chart(fig_data, title, names, colors):
    fig = go.Figure()
    for key, name, color in zip(fig_data, names, colors):
        fig.add_scatter(y=key, x=list(range(1, len(key) + 1)), name=name, mode="lines+markers", line=dict(color=color, width=3))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=MUTED)), height=300, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, family="Inter"),
        xaxis=dict(title="Epoch", gridcolor=LINE, dtick=1), yaxis=dict(gridcolor=LINE),
        legend=dict(orientation="h", y=1.15, x=1, xanchor="right"),
    )
    return fig


# ---------------------------------------------------------------- state
st.session_state.setdefault("seed", EXAMPLES["Plot twist"])  # initial value of the input widget
st.session_state.setdefault("ver", 0)


def set_example(label):
    st.session_state.seed = EXAMPLES[label]
    st.session_state.ver += 1  # new widget key -> input resets to the example


model, vocab, metrics = load_assets()

# ---------------------------------------------------------------- layout
st.markdown(
    '<div class="hero"><h1>SentiScope</h1>'
    "<p>A neural network that reads your text, scores its sentiment and shows exactly which words changed its mind.</p></div>",
    unsafe_allow_html=True,
)
tab_a, tab_b, tab_c = st.tabs(["Analyze", "Model insights", "How it works"])

with tab_a:
    ex_cols = st.columns(len(EXAMPLES))
    for col, label in zip(ex_cols, EXAMPLES):
        col.button(label, key=f"ex_{label}", on_click=set_example, args=(label,), use_container_width=True)

    modes = ["Live typing", "Paste a review"] if st_keyup else ["Paste a review"]
    mode = st.radio("Input mode", modes, horizontal=True, label_visibility="collapsed")
    ver = st.session_state.ver
    if mode == "Live typing":
        text = st_keyup("Type a sentence - the analysis updates as you type", value=st.session_state.seed,
                        key=f"live_{ver}", debounce=250) or ""
    else:
        text = st.text_area("Paste a review", value=st.session_state.seed, height=130, key=f"area_{ver}")

    res = analyse(text, model, vocab)
    if res is None:
        st.info("Start typing to see the network react.")
    else:
        label, color = verdict(res["p"])
        left, right = st.columns([1, 1.35], gap="large")
        with left:
            with st.container(border=True):
                st.markdown('<div class="ptitle">Sentiment score</div>', unsafe_allow_html=True)
                st.plotly_chart(gauge(res["p"]), use_container_width=True, config={"displayModeBar": False})
                known = sum(1 for w in res["words"] if w in vocab)
                st.markdown(
                    f'<span class="verdict" style="background:{color}22;color:{color}">{label}</span>'
                    f'<span style="color:{MUTED};margin-left:.8rem;font-size:.9rem">'
                    f'{len(res["words"])} words, {known} recognised by the model</span>',
                    unsafe_allow_html=True,
                )
        with right:
            card("Word-level explanation", highlighted(res) +
                 f'<div class="legend"><span style="color:{POS}">Teal</span> pushes towards positive, '
                 f'<span style="color:{NEG}">coral</span> towards negative. Hover a word for its exact weight.</div>')
        st.markdown("&nbsp;")
        c1, c2 = st.columns([1.35, 1], gap="large")
        with c1:
            with st.container(border=True):
                st.markdown('<div class="ptitle">Most influential words</div>', unsafe_allow_html=True)
                st.plotly_chart(influence_chart(res), use_container_width=True, config={"displayModeBar": False})
        with c2:
            card("Reading the result",
                 "<small style='color:#8B97AD;line-height:1.7'>Each word is removed in turn and the network is run again. "
                 "The shift in its output is that word's influence. This is called <b>occlusion analysis</b> and works "
                 "with any model, so the explanation reflects the network's real behaviour, not a guess.</small>")

with tab_b:
    m = metrics
    cols = st.columns(4)
    metric(cols[0], f"{m['test_acc'] * 100:.1f}%", "Test accuracy")
    metric(cols[1], f"{m['params']:,}", "Parameters")
    metric(cols[2], f"{m['train_size'] + m['test_size']:,}", "Reviews in dataset")
    metric(cols[3], str(m["epochs"]), "Epochs trained")
    st.markdown("&nbsp;")
    h = m["history"]
    g1, g2 = st.columns(2, gap="large")
    g1.plotly_chart(chart([h["loss"], h["val_loss"]], "Loss", ["Training", "Validation"], [POS, AMB]),
                    use_container_width=True, config={"displayModeBar": False})
    g2.plotly_chart(chart([h["accuracy"], h["val_accuracy"]], "Accuracy", ["Training", "Validation"], [POS, AMB]),
                    use_container_width=True, config={"displayModeBar": False})
    a, b = st.columns([1.3, 1], gap="large")
    with a:
        st.markdown("##### Architecture")
        st.dataframe(pd.DataFrame(m["layers"], columns=["Layer", "Type", "Output shape", "Parameters"]),
                     hide_index=True, use_container_width=True)
    with b:
        st.markdown("##### Confusion matrix (test set)")
        cm = m["confusion"]
        fig = go.Figure(go.Heatmap(z=cm, x=["Pred. negative", "Pred. positive"], y=["Actual negative", "Actual positive"],
                                   text=[[f"{v:,}" for v in r] for r in cm], texttemplate="%{text}",
                                   colorscale=[[0, CARD], [1, POS]], showscale=False))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=TEXT, family="Inter"), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with tab_c:
    steps = [
        ("01", "Tokenise", "Text is lower-cased and split into words."),
        ("02", "Index", "Each word becomes an integer from a 10,000-word vocabulary."),
        ("03", "Embed", "Integers map to 64-number vectors that capture meaning."),
        ("04", "Bi-LSTM", "Reads the sentence forwards and backwards, keeping context."),
        ("05", "Dense layers", "Combine the features and learn the decision."),
        ("06", "Sigmoid", "Outputs a probability between 0 (negative) and 1 (positive)."),
    ]
    cols = st.columns(3)
    for i, (n, t, d) in enumerate(steps):
        cols[i % 3].markdown(f'<div class="step"><b>{n}</b><div>{t}</div><small>{d}</small></div>', unsafe_allow_html=True)
        if i == 2:
            st.markdown("&nbsp;")
    st.markdown("&nbsp;")
    l, r = st.columns(2, gap="large")
    with l:
        card("Training", "<small style='color:#8B97AD;line-height:1.7'>Trained on the IMDB Movie Reviews dataset "
             "(25,000 training and 25,000 test reviews) with the Adam optimiser, binary cross-entropy loss, dropout "
             "regularisation and early stopping on validation loss.</small>")
    with r:
        card("Good to know", "<small style='color:#8B97AD;line-height:1.7'>The model learned from English movie reviews, so "
             "it is most reliable on opinionated text about films, products and services. Sarcasm and domain "
             "jargon can still fool it.</small>")
st.markdown(f"<p style='text-align:center;color:{MUTED};font-size:.8rem;margin-top:2.5rem'>"
            "SentiScope - built with TensorFlow, Keras and Streamlit</p>", unsafe_allow_html=True)
