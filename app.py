import difflib
import re
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


ALIASES = {
    "js": "javascript", "node": "node.js", "nodejs": "node.js", "reactjs": "react", "react.js": "react",
    "ml": "machine learning", "dl": "deep learning", "ai": "llm", "py": "python", "golang": "go",
    "cpp": "c++", "csharp": "c#", "dotnet": "net", "k8s": "kubernetes", "postgres": "postgresql",
    "mongo": "mongodb", "tf": "tensorflow", "sklearn": "scikit-learn", "powerbi": "power bi",
    "ts": "typescript", "springboot": "spring boot", "ci cd": "ci/cd", "cicd": "ci/cd", "rest": "rest api",
}
TIME_OPTIONS = {"Kabhi bhi": None, "Last 24 hours": 1, "Last 3 days": 3, "Last 7 days": 7, "Last 30 days": 30}


@st.cache_data
def load_jobs():
    base = Path(__file__).parent
    path = base / "jobs.csv"
    if not path.exists():
        path = base / "data" / "jobs.csv"
    df = pd.read_csv(path)
    df["skill_list"] = df["skills"].apply(lambda s: [x.strip() for x in str(s).split(";") if x.strip()])
    df["text"] = df["title"] + " " + df["skills"].str.replace(";", " ") + " " + df["description"]
    if "posted_date" in df.columns:
        df["posted"] = pd.to_datetime(df["posted_date"], errors="coerce")
    else:
        df["posted"] = pd.Timestamp.today().normalize()
    df["posted"] = df["posted"].fillna(pd.Timestamp.today().normalize())
    df["days_ago"] = (pd.Timestamp.today().normalize() - df["posted"]).dt.days.clip(lower=0)
    return df


@st.cache_resource
def get_vectorizer(texts: tuple):
    vec = TfidfVectorizer(stop_words="english", token_pattern=r"[A-Za-z0-9+#.]+", lowercase=True)
    return vec, vec.fit_transform(texts)


def split_input(txt):
    return [p.strip() for p in re.split(r"[,;\n]+", txt or "") if p.strip()]


def resolve_skills(raw, known):
    """Typed/selected skills -> canonical names. Handles case, aliases and small typos."""
    lookup = {k.lower(): k for k in known}
    found, unknown = [], []
    for s in raw:
        key = ALIASES.get(s.lower().strip(), s.lower().strip())
        if key in lookup:
            hit = lookup[key]
        else:
            m = difflib.get_close_matches(key, list(lookup), n=1, cutoff=0.75)
            hit = lookup[m[0]] if m else None
        if hit:
            if hit not in found:
                found.append(hit)
        else:
            unknown.append(s)
    return found, unknown


def recommend(df, user_skills, role, exp, locs, types, min_sal, max_days, sort_by):
    mask = (df.experience <= exp + 1) & (df.salary_lpa >= min_sal)
    if max_days:
        mask &= df.days_ago <= max_days
    if locs:  # partial, case-insensitive match: "pune", "Delhi NCR" etc.
        pat = "|".join(re.escape(l.lower()) for l in locs)
        mask &= df.location.str.lower().str.contains(pat, regex=True)
    if types:
        mask &= df.type.isin(types)
    d = df[mask].copy()
    if d.empty:
        return d.assign(score=0, matched=[], missing=[])

    role = (role or "").strip()
    us = {s.lower() for s in user_skills}
    if us or role:
        vec, mat_all = get_vectorizer(tuple(df["text"]))
        q = vec.transform([" ".join(user_skills) + " " + role])
        sim = cosine_similarity(q, mat_all[[df.index.get_loc(i) for i in d.index]]).flatten()
        sim = sim / sim.max() if sim.max() > 0 else sim
    else:
        sim = pd.Series(0.0, index=d.index).values

    d["matched"] = d.skill_list.apply(lambda l: [s for s in l if s.lower() in us])
    d["missing"] = d.skill_list.apply(lambda l: [s for s in l if s.lower() not in us])
    overlap = d.matched.apply(len) / d.skill_list.apply(len)
    d["score"] = ((0.5 * sim + 0.5 * overlap) * 100).round().astype(int)
    if us or role:
        d = d[d.score > 0]

    if sort_by == "Newest first":
        return d.sort_values(["days_ago", "score"], ascending=[True, False])
    if sort_by == "Highest salary":
        return d.sort_values("salary_lpa", ascending=False)
    return d.sort_values(["score", "days_ago"], ascending=[False, True])


def ago(n):
    return "Today" if n == 0 else "Yesterday" if n == 1 else f"{n} days ago"


def chips(items, cls):
    return "".join(f"<span class='chip {cls}'>{i}</span>" for i in items)


def card(r, key, show_score=True):
    badge = f"{r.score}% match" if show_score else ago(r.days_ago)
    st.markdown(f"""
<div class='card'>
  <span class='badge'>{badge}</span>
  <p class='title'>{r.title}</p>
  <p class='meta'>🏢 {r.company} &nbsp;•&nbsp; 📍 {r.location} &nbsp;•&nbsp; 🕒 {r.type}
  &nbsp;•&nbsp; 💰 ₹{r.salary_lpa} LPA &nbsp;•&nbsp; 🎯 {r.experience}+ yrs &nbsp;•&nbsp; 🗓️ {ago(r.days_ago)}</p>
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
    picked = st.multiselect("Skills / Languages", all_skills, default=["Python", "SQL"])
    typed = st.text_input("Aur skills type karo (comma se)", placeholder="e.g. js, kotlin, flutter, pyhton")
    role = st.text_input("Target role (optional)", placeholder="e.g. Data Analyst")
    exp = st.slider("Experience (years)", 0, 8, 0)

    st.header("🔎 Filters")
    all_locs = sorted(df.location.unique())
    locs = st.multiselect("Location", all_locs)
    typed_loc = st.text_input("Ya location type karo", placeholder="e.g. pune, delhi, remote")
    posted_label = st.selectbox("Posted within", list(TIME_OPTIONS), index=3)
    types = st.multiselect("Job type", sorted(df.type.unique()))
    min_sal = st.slider("Min salary (LPA)", 0, 40, 0)
    sort_by = st.radio("Sort by", ["Best match", "Newest first", "Highest salary"], horizontal=False)

skills, unknown = resolve_skills(picked + split_input(typed), all_skills)
all_locations = locs + split_input(typed_loc)
if unknown:
    st.toast(f"Dataset mein nahi mili: {', '.join(unknown)}", icon="⚠️")

results = recommend(df, skills, role, exp, all_locations, types, min_sal,
                    TIME_OPTIONS[posted_label], sort_by)
if "limit" not in st.session_state:
    st.session_state.limit = 10
sig = (tuple(skills), role, exp, tuple(all_locations), posted_label, tuple(types), min_sal, sort_by)
if st.session_state.get("sig") != sig:  # filters changed -> reset pagination
    st.session_state.sig = sig
    st.session_state.limit = 10

tab1, tab2, tab3 = st.tabs(["✨ Recommendations", "★ Saved Jobs", "📊 Market Insights"])

with tab1:
    st.caption(f"📦 Dataset: {len(df)} jobs • latest posting: {df.posted.max():%d %b %Y}")
    if skills:
        st.write("**Aapki skills:** " + chips(skills, "have"), unsafe_allow_html=True)
    if results.empty:
        st.warning("Koi job match nahi hui. Posted-within, location ya experience filter relax karke dekho.")
    else:
        has_query = bool(skills) or bool(role.strip())
        if not has_query:
            st.info("Skills select/type karo to match % dikhega. Abhi latest jobs dikh rahi hain.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Jobs found", len(results))
        c2.metric("Best match" if has_query else "Newest", f"{results.score.max()}%" if has_query
                  else ago(results.days_ago.min()))
        gaps = pd.Series([s for l in results.head(5).missing for s in l])
        c3.metric("Top skill to learn", gaps.value_counts().index[0] if len(gaps) and has_query else "—")
        st.write("")
        for idx, r in results.head(st.session_state.limit).iterrows():
            card(r, idx, show_score=has_query)
        if len(results) > st.session_state.limit:
            if st.button(f"⬇️ Aur dikhao ({len(results) - st.session_state.limit} baaki)"):
                st.session_state.limit += 10
                st.rerun()
        st.download_button("📥 Results CSV download", results[["title", "company", "location", "type",
                           "experience", "salary_lpa", "skills", "posted_date"]].to_csv(index=False),
                           "recommended_jobs.csv", "text/csv")

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
