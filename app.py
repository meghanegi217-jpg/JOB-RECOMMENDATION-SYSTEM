from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="JobMatch", page_icon="💼", layout="wide")

st.markdown("""
<style>
.hero{padding:28px 32px;border-radius:18px;margin-bottom:18px;color:#fff;
 background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 55%,#ec4899 100%);}
.hero h1{margin:0;font-size:2.1rem;color:#fff}
.hero p{margin:6px 0 0;opacity:.92}
.card{border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:18px 20px;
 margin-bottom:6px;background:rgba(128,128,128,.06);transition:.2s}
.card:hover{border-color:#7c3aed;box-shadow:0 6px 22px rgba(124,58,237,.18)}
.title{font-size:1.2rem;font-weight:700;margin:0}
.meta{opacity:.75;font-size:.9rem;margin:2px 0 10px}
.chip{display:inline-block;padding:3px 11px;margin:2px 4px 2px 0;border-radius:999px;font-size:.78rem}
.have{background:#dcfce7;color:#166534}
.miss{background:#fee2e2;color:#991b1b}
.badge{float:right;padding:6px 14px;border-radius:12px;font-weight:700;color:#fff;
 background:linear-gradient(135deg,#4f46e5,#7c3aed)}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_jobs():
    base = Path(__file__).parent
    path = base / "jobs.csv"
    if not path.exists():
        path = base / "data" / "jobs.csv"
    df = pd.read_csv(path)
    df["skill_list"] = df["skills"].apply(lambda s: [x.strip() for x in s.split(";")])
    df["text"] = df["title"] + " " + df["skills"].str.replace(";", " ") + " " + df["description"]
    return df


def recommend(df, user_skills, role, exp, locs, types, min_sal):
    d = df[(df.experience <= exp + 1) & (df.salary_lpa >= min_sal)].copy()
    if locs:
        d = d[d.location.isin(locs)]
    if types:
        d = d[d.type.isin(types)]
    if d.empty or not user_skills:
        return d.iloc[0:0]

    query = " ".join(user_skills) + " " + role
    mat = TfidfVectorizer(stop_words="english").fit_transform(d["text"].tolist() + [query])
    sim = cosine_similarity(mat[-1], mat[:-1]).flatten()
    sim = sim / sim.max() if sim.max() > 0 else sim

    us = {s.lower() for s in user_skills}
    d["matched"] = d.skill_list.apply(lambda l: [s for s in l if s.lower() in us])
    d["missing"] = d.skill_list.apply(lambda l: [s for s in l if s.lower() not in us])
    overlap = d.matched.apply(len) / d.skill_list.apply(len)
    d["score"] = ((0.5 * sim + 0.5 * overlap) * 100).round().astype(int)
    return d[d.score > 0].sort_values("score", ascending=False)


def chips(items, cls):
    return "".join(f"<span class='chip {cls}'>{i}</span>" for i in items)


def card(r, key):
    st.markdown(f"""
<div class='card'>
  <span class='badge'>{r.score}% match</span>
  <p class='title'>{r.title}</p>
  <p class='meta'>🏢 {r.company} &nbsp;•&nbsp; 📍 {r.location} &nbsp;•&nbsp; 🕒 {r.type}
  &nbsp;•&nbsp; 💰 ₹{r.salary_lpa} LPA &nbsp;•&nbsp; 🎯 {r.experience}+ yrs</p>
  <p>{r.description}</p>
  <div>{chips(r.matched, 'have')}{chips(r.missing, 'miss')}</div>
</div>""", unsafe_allow_html=True)
    saved = key in st.session_state.saved
    if st.button("★ Saved" if saved else "☆ Save job", key=f"s{key}"):
        st.session_state.saved.discard(key) if saved else st.session_state.saved.add(key)
        st.rerun()


df = load_jobs()
if "saved" not in st.session_state:
    st.session_state.saved = set()

st.markdown("""<div class='hero'><h1>💼 JobMatch</h1>
<p>Apni skills batao — hum tumhare liye best jobs dhundhenge, skill gap ke saath.</p></div>""",
            unsafe_allow_html=True)

with st.sidebar:
    st.header("👤 Your Profile")
    all_skills = sorted({s for l in df.skill_list for s in l})
    skills = st.multiselect("Skills", all_skills, default=["Python", "SQL"])
    role = st.text_input("Target role (optional)", placeholder="e.g. Data Analyst")
    exp = st.slider("Experience (years)", 0, 8, 0)
    st.header("🔎 Filters")
    locs = st.multiselect("Location", sorted(df.location.unique()))
    types = st.multiselect("Job type", sorted(df.type.unique()))
    min_sal = st.slider("Min salary (LPA)", 0, 20, 0)

results = recommend(df, skills, role, exp, locs, types, min_sal)

tab1, tab2, tab3 = st.tabs(["✨ Recommendations", "★ Saved Jobs", "📊 Market Insights"])

with tab1:
    if not skills:
        st.info("Sidebar se apni skills select karo.")
    elif results.empty:
        st.warning("Koi job match nahi hui. Filters thode relax karo.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Jobs found", len(results))
        c2.metric("Best match", f"{results.score.iloc[0]}%")
        gaps = pd.Series([s for l in results.head(5).missing for s in l])
        c3.metric("Top skill to learn", gaps.value_counts().index[0] if len(gaps) else "—")
        st.write("")
        for idx, r in results.head(10).iterrows():
            card(r, idx)

with tab2:
    if not st.session_state.saved:
        st.info("Abhi koi job save nahi ki.")
    for idx in sorted(st.session_state.saved):
        r = df.loc[idx]
        st.markdown(f"**{r.title}** — {r.company}, {r.location} (₹{r.salary_lpa} LPA)")

with tab3:
    a, b = st.columns(2)
    a.subheader("Most in-demand skills")
    a.bar_chart(pd.Series([s for l in df.skill_list for s in l]).value_counts().head(10))
    b.subheader("Jobs by location")
    b.bar_chart(df.location.value_counts())
    st.subheader("Avg salary by job type (LPA)")
    st.bar_chart(df.groupby("type").salary_lpa.mean().round(1))
