"""BenchLab — comparativo global de algoritmos x datasets de saude."""
import random
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from PIL import Image as _PILImage
import streamlit as st

from core.benchlab import BENCH_ALGORITHMS, load_cache, run_benchlab, save_cache
from core.data.benchmarks import BENCHMARKS

_favicon = _PILImage.open(Path(__file__).parent.parent / "favicon.png")
st.set_page_config(
    page_title="BenchLab — Lab AI Prediction",
    page_icon=_favicon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20,300,0,0" />
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" />

<style>
header, footer,
[data-testid="stSidebar"], [data-testid="stSidebarNav"],
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], #MainMenu { display: none !important; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .block-container {
    background-color: #ffffff !important; color: #111827 !important;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
.block-container { padding: 3rem 3rem !important; max-width: 1200px !important; }

.ms { font-family: 'Material Symbols Outlined'; font-style: normal; font-weight: normal;
    font-size: 1rem; line-height: 1; vertical-align: middle; display: inline-block; color: #111827; }
.ms-lg { font-size: 1.4rem; margin-right: .25rem; }

h1,h2,h3,h4,h5,h6 { font-family:"Space Grotesk","Inter",sans-serif !important; letter-spacing:-.01em; }

.bl-title { font-family:"Space Grotesk","Inter",sans-serif; font-size:1.6rem; font-weight:700;
    color:#223886 !important; margin-bottom:.15rem; display:flex; align-items:center; gap:.35rem; }
.bl-sub { font-size:.9rem; color:#6b7280 !important; margin-bottom:1.5rem; max-width:700px; line-height:1.55; }
.bl-section { font-size:.68rem; font-weight:700; color:#9ca3af !important; text-transform:uppercase;
    letter-spacing:.1em; margin:1.75rem 0 .5rem; padding-bottom:.35rem; border-bottom:1px solid #f3f4f6; }
.bl-metric-label { font-size:.78rem; font-weight:600; color:#374151 !important; margin-bottom:.3rem; }
.bl-note { font-size:.78rem; color:#9ca3af !important; margin:.5rem 0 1.5rem; line-height:1.5; }
.bl-error-row { font-size:.78rem; color:#dc2626 !important; padding:.25rem 0; }

div[data-testid="stButton"] { display:flex !important; justify-content:flex-start !important; }
div[data-testid="stButton"] > button {
    background-color:#ffffff !important; color:#111827 !important;
    border:1px solid #e5e7eb !important; border-radius:6px !important;
    font-size:.8rem !important; font-weight:500 !important;
    padding:5px 16px !important; width:auto !important; transition:all .12s !important; }
div[data-testid="stButton"] > button:hover {
    background-color:#eef1fb !important; border-color:#223886 !important; }
div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
    background-color:#223886 !important; color:#ffffff !important;
    border-color:#223886 !important; font-weight:600 !important;
    border-bottom:3px solid #9ec83b !important; }
div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {
    background-color:#1a2b66 !important; border-color:#1a2b66 !important; }
</style>
""", unsafe_allow_html=True)

# ── Navegacao ─────────────────────────────────────────────────────────────────
_back_col, _ = st.columns([1, 9])
with _back_col:
    if st.button("← Voltar", key="bl_back", type="secondary"):
        st.switch_page("app.py")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="bl-title"><span class="ms">bolt</span>BENCHLAB</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="bl-sub">'
    'Compare todos os algoritmos em todos os datasets de uma vez. '
    'Cada combinacao roda com validacao cruzada (5 folds) e parametros default, '
    'sem HPO, para uma comparacao justa de base.'
    '</p>',
    unsafe_allow_html=True,
)

# ── Datasets e algoritmos disponiveis ─────────────────────────────────────────
OK_BENCHES = [b for b in BENCHMARKS.values() if b.status == "ok" and b.loader is not None]
INTL_BENCHES = [b for b in OK_BENCHES if b.country != "br"]
BR_BENCHES = [b for b in OK_BENCHES if b.country == "br"]
ALGO_DISPLAY = {v: k for k, v in BENCH_ALGORITHMS.items()}  # key -> nome

# ── Configuracao ──────────────────────────────────────────────────────────────
st.markdown('<p class="bl-section">Configurar</p>', unsafe_allow_html=True)

cfg_col1, cfg_col2, cfg_col3 = st.columns([3, 3, 2])

with cfg_col1:
    st.markdown('<p class="bl-metric-label">Datasets</p>', unsafe_allow_html=True)
    selected_datasets = []
    for bench in INTL_BENCHES:
        checked = st.checkbox(bench.name, value=True, key=f"bl_ds_{bench.key}")
        if checked:
            selected_datasets.append(bench)

    # ── Datasets brasileiros (DATASUS), com sorteio entre os disponiveis ───────
    if BR_BENCHES:
        st.markdown(
            '<p class="bl-metric-label" style="margin-top:.85rem">Brasil (DATASUS)</p>',
            unsafe_allow_html=True,
        )
        include_br = st.checkbox(
            "Incluir dataset brasileiro", value=False, key="bl_br_on"
        )
        if include_br:
            br_by_key = {b.key: b for b in BR_BENCHES}
            mode = st.radio(
                "Selecao do dataset brasileiro",
                ["Aleatorio", "Escolher"],
                horizontal=True,
                key="bl_br_mode",
                label_visibility="collapsed",
            )
            if mode == "Aleatorio":
                k = st.number_input(
                    "Quantos sortear",
                    min_value=1, max_value=len(BR_BENCHES), value=1, step=1,
                    key="bl_br_k",
                )
                draw = st.button("Sortear", key="bl_br_draw", type="secondary")
                cur = st.session_state.get("bl_br_pick", [])
                # (re)sorteia ao clicar, ao habilitar pela 1a vez, ou se a
                # quantidade muda; senao mantem o sorteio estavel entre reruns.
                stale = (not cur) or len(cur) != int(k) or any(x not in br_by_key for x in cur)
                if draw or stale:
                    cur = [b.key for b in random.sample(BR_BENCHES, int(k))]
                    st.session_state["bl_br_pick"] = cur
                picked = [br_by_key[x] for x in cur if x in br_by_key]
                if picked:
                    names = ", ".join(b.name for b in picked)
                    st.markdown(
                        f'<p class="bl-note">Sorteado: <b>{names}</b></p>',
                        unsafe_allow_html=True,
                    )
                selected_datasets += picked
            else:
                chosen = st.multiselect(
                    "Datasets brasileiros",
                    [b.name for b in BR_BENCHES],
                    key="bl_br_choose",
                    label_visibility="collapsed",
                )
                selected_datasets += [b for b in BR_BENCHES if b.name in chosen]

with cfg_col2:
    st.markdown('<p class="bl-metric-label">Algoritmos</p>', unsafe_allow_html=True)
    selected_algos = []
    for algo_key, algo_name in ALGO_DISPLAY.items():
        checked = st.checkbox(algo_name, value=True, key=f"bl_algo_{algo_key}")
        if checked:
            selected_algos.append(algo_key)

with cfg_col3:
    st.markdown('<p class="bl-metric-label">Folds de CV</p>', unsafe_allow_html=True)
    n_folds = st.selectbox("", [3, 5, 10], index=1, key="bl_folds", label_visibility="collapsed")
    st.markdown(
        '<p class="bl-note">'
        'Params default, sem HPO. '
        'Tempo estimado: ~30-90 s para 4 datasets x 6 modelos.'
        '</p>',
        unsafe_allow_html=True,
    )

st.markdown("")
run_col, rerun_col, _ = st.columns([2, 2, 6])

cached = load_cache()

with run_col:
    run_btn = st.button(
        "Rodar BenchLab",
        key="bl_run",
        type="primary",
        disabled=not selected_datasets or not selected_algos,
    )
with rerun_col:
    if cached:
        rerun_btn = st.button("Re-rodar", key="bl_rerun", type="secondary")
    else:
        rerun_btn = False

# ── Executar ──────────────────────────────────────────────────────────────────
if run_btn or rerun_btn:
    prog_bar = st.progress(0.0)
    status_txt = st.empty()

    def _cb(msg: str, frac: float) -> None:
        prog_bar.progress(min(frac, 1.0))
        status_txt.markdown(f'<p style="font-size:.82rem;color:#6b7280">{msg}</p>', unsafe_allow_html=True)

    result = run_benchlab(
        benchmarks=selected_datasets,
        algorithm_keys=selected_algos,
        n_folds=int(n_folds),
        progress_cb=_cb,
    )
    save_cache(result)
    cached = result
    prog_bar.empty()
    status_txt.empty()
    st.success("Benchmark concluido e salvo.")

# ── Resultados ────────────────────────────────────────────────────────────────
if not cached:
    st.markdown(
        '<p class="bl-note" style="margin-top:2rem">'
        'Nenhum resultado ainda. Configure e clique em <b>Rodar BenchLab</b>.'
        '</p>',
        unsafe_allow_html=True,
    )
    st.stop()

results = cached["results"]
meta = cached.get("metadata", {})
errors = cached.get("errors", [])

st.markdown('<p class="bl-section">Resultados</p>', unsafe_allow_html=True)

# Seletor de metrica
metric_options = {
    "AUROC": "roc_auc",
    "PR-AUC": "pr_auc",
    "F1": "f1",
    "Precisao": "precision",
    "Recall / Sensibilidade": "recall",
    "Especificidade": "specificity",
    "Brier Score (menor = melhor)": "brier",
}
met_col, _ = st.columns([3, 7])
with met_col:
    metric_label = st.selectbox(
        "Metrica exibida",
        list(metric_options.keys()),
        index=0,
        key="bl_metric",
    )
metric_key = metric_options[metric_label]
lower_is_better = metric_key == "brier"

# Monta DataFrame de resultados
bench_names = {b.key: b.name for b in OK_BENCHES}
present_benches = [k for k in results if results[k]]
present_algos = sorted({ak for row in results.values() for ak in row})

rows = []
for bk in present_benches:
    row = {"Dataset": bench_names.get(bk, bk)}
    for ak in present_algos:
        val = results[bk].get(ak, {}).get(metric_key)
        row[ALGO_DISPLAY.get(ak, ak)] = round(val, 4) if val is not None else None
    rows.append(row)

df_res = pd.DataFrame(rows).set_index("Dataset")
algo_cols = [c for c in df_res.columns]

# ── Heatmap ───────────────────────────────────────────────────────────────────
z_vals = df_res.values.tolist()
text_vals = [
    [f"{v:.3f}" if v is not None else "err" for v in row]
    for row in z_vals
]

# Marca o melhor de cada dataset (linha)
best_text = []
for i, row in enumerate(z_vals):
    valid = [(j, v) for j, v in enumerate(row) if v is not None]
    if valid:
        if lower_is_better:
            best_j = min(valid, key=lambda x: x[1])[0]
        else:
            best_j = max(valid, key=lambda x: x[1])[0]
    else:
        best_j = -1
    best_text.append([
        f"<b>{text_vals[i][j]}</b>" if j == best_j else text_vals[i][j]
        for j in range(len(row))
    ])

colorscale = [
    [0.0, "#f0f4ff"],
    [0.5, "#8fa8e8"],
    [1.0, "#223886"],
] if not lower_is_better else [
    [0.0, "#223886"],
    [0.5, "#8fa8e8"],
    [1.0, "#f0f4ff"],
]

fig_heat = go.Figure(go.Heatmap(
    z=z_vals,
    x=algo_cols,
    y=list(df_res.index),
    text=best_text,
    texttemplate="%{text}",
    textfont={"size": 13, "family": "Inter, sans-serif"},
    colorscale=colorscale,
    zmin=0.5 if not lower_is_better else 0.0,
    zmax=1.0 if not lower_is_better else 0.5,
    showscale=True,
    colorbar=dict(
        title=dict(text=metric_label, font=dict(size=11, family="Inter, sans-serif")),
        tickfont=dict(size=10, family="Inter, sans-serif"),
        thickness=12, len=0.8,
    ),
    hovertemplate="<b>%{y}</b><br>%{x}<br>" + metric_label + ": %{z:.4f}<extra></extra>",
))
fig_heat.update_layout(
    height=max(280, 90 * len(present_benches)),
    margin=dict(l=0, r=40, t=20, b=20),
    font=dict(family="Inter, sans-serif", size=12),
    xaxis=dict(side="top", tickfont=dict(size=12, family="Inter, sans-serif")),
    yaxis=dict(tickfont=dict(size=12, family="Inter, sans-serif"), autorange="reversed"),
    paper_bgcolor="white",
    plot_bgcolor="white",
)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown(
    '<p class="bl-note">Valores em <b>negrito</b> = melhor por dataset (linha). '
    f'CV com {meta.get("n_folds", 5)} folds, params default, sem HPO.</p>',
    unsafe_allow_html=True,
)

# ── Ranking por vitorias ──────────────────────────────────────────────────────
st.markdown('<p class="bl-section">Ranking por vitorias</p>', unsafe_allow_html=True)

wins: dict[str, int] = {col: 0 for col in algo_cols}
for row in z_vals:
    valid = [(j, v) for j, v in enumerate(row) if v is not None]
    if not valid:
        continue
    if lower_is_better:
        best_j = min(valid, key=lambda x: x[1])[0]
    else:
        best_j = max(valid, key=lambda x: x[1])[0]
    wins[algo_cols[best_j]] += 1

wins_sorted = sorted(wins.items(), key=lambda x: x[1], reverse=True)
win_names = [w[0] for w in wins_sorted]
win_vals = [w[1] for w in wins_sorted]

fig_rank = go.Figure(go.Bar(
    x=win_vals,
    y=win_names,
    orientation="h",
    marker_color=["#223886" if i == 0 else "#8fa8e8" if i == 1 else "#c7d4f5"
                  for i in range(len(win_names))],
    text=[str(v) for v in win_vals],
    textposition="outside",
    textfont=dict(size=13, family="Inter, sans-serif", color="#111827"),
    hovertemplate="%{y}: %{x} vitoria(s)<extra></extra>",
))
fig_rank.update_layout(
    height=max(180, 52 * len(win_names)),
    margin=dict(l=0, r=60, t=10, b=10),
    xaxis=dict(
        title=dict(text="Vitorias (melhor " + metric_label + " por dataset)", font=dict(size=11)),
        tickfont=dict(size=11, family="Inter, sans-serif"),
        gridcolor="#f3f4f6",
    ),
    yaxis=dict(
        tickfont=dict(size=12, family="Inter, sans-serif"),
        autorange="reversed",
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
)
st.plotly_chart(fig_rank, use_container_width=True)

# ── Tabela numerica ───────────────────────────────────────────────────────────
with st.expander("Ver tabela numerica completa"):
    st.dataframe(df_res.style.format("{:.4f}", na_rep="err"), use_container_width=True)

# ── Erros ─────────────────────────────────────────────────────────────────────
if errors:
    with st.expander(f"Erros ({len(errors)})"):
        for bk, ak, msg in errors:
            st.markdown(
                f'<p class="bl-error-row">'
                f'<b>{bench_names.get(bk, bk)}</b> x <b>{ALGO_DISPLAY.get(ak, ak)}</b>: {msg}'
                f'</p>',
                unsafe_allow_html=True,
            )

# ── Rodape ────────────────────────────────────────────────────────────────────
st.markdown(
    '<hr style="border:none;border-top:1px solid #f3f4f6;margin:3rem 0 1rem">',
    unsafe_allow_html=True,
)
