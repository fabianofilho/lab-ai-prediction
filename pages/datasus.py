"""Lab AI Prediction — Catálogo de desfechos DATASUS."""
from pathlib import Path
from PIL import Image as _PILImage
import streamlit as st

_favicon = _PILImage.open(Path(__file__).parent.parent / "favicon.png")
st.set_page_config(
    page_title="DATASUS — Lab AI Prediction",
    page_icon=_favicon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Metadados dos desfechos ────────────────────────────────────────────────────
OUTCOME_GROUPS = {
    "Saúde Materno-Infantil": [
        {"key": "baixo_peso_nascer", "icon": "scale", "name": "Baixo Peso ao Nascer",
         "source": "SINASC", "est_min": 5, "status": "ok", "linkage": None,
         "note": "Base única SINASC. Pipeline completo disponível."},
        {"key": "prematuridade", "icon": "baby_changing_station", "name": "Prematuridade",
         "source": "SINASC", "est_min": 5, "status": "ok", "linkage": None,
         "note": "Base única SINASC. Pipeline completo disponível."},
        {"key": "apgar_baixo", "icon": "cardiology", "name": "Apgar Baixo no 5º Minuto",
         "source": "SINASC", "est_min": 5, "status": "ok", "linkage": None,
         "note": "Base única SINASC. Pipeline completo disponível."},
        {"key": "cesarea", "icon": "pregnant_woman", "name": "Parto Cesariana",
         "source": "SINASC", "est_min": 5, "status": "ok", "linkage": None,
         "note": "Base única SINASC. Taxa de cesárea no Brasil entre as maiores do mundo."},
        {"key": "anomalia_congenita", "icon": "neurology", "name": "Anomalia Congênita",
         "source": "SINASC", "est_min": 5, "status": "ok", "linkage": None,
         "note": "Base única SINASC. Desfecho raro (~0,9%), considere balanceamento."},
        {"key": "mortalidade_neonatal", "icon": "child_care", "name": "Mortalidade Neonatal",
         "source": "SINASC + SIM", "est_min": 12, "status": "ok",
         "linkage": "Linkage determinístico SINASC ↔ SIM por DTNASC/SEXO/PESO. Pareamento aproximado.",
         "note": "Linkage entre nascimentos (SINASC) e óbitos neonatais (SIM). Prevalência baixa, considere balanceamento."},
    ],
    "Internação Hospitalar": [
        {"key": "readmissao_30d", "icon": "sync", "name": "Readmissão Hospitalar 30 dias",
         "source": "SIH", "est_min": 10, "status": "ok",
         "linkage": "Linkage probabilístico por quase-identificadores (nascimento + sexo + município + CEP), pois o SIH-RD público não traz CNS/CPF.",
         "note": "Reinternação em 30 dias para o mesmo paciente. Use amostra grande (≥ 30.000) para taxa representativa."},
        {"key": "permanencia_prolongada", "icon": "bed", "name": "Permanência Hospitalar Prolongada",
         "source": "SIH", "est_min": 10, "status": "ok", "linkage": None,
         "note": "Base única SIH. Pipeline completo disponível."},
        {"key": "uso_uti", "icon": "vital_signs", "name": "Uso de UTI na Internação",
         "source": "SIH", "est_min": 10, "status": "ok", "linkage": None,
         "note": "Base única SIH. Alvo derivado dos dias de UTI (UTI_MES_TO)."},
        {"key": "infeccao_hospitalar", "icon": "coronavirus", "name": "Infecção Hospitalar",
         "source": "SIH", "est_min": 10, "status": "ok", "linkage": None,
         "note": "Base única SIH. Pipeline completo disponível."},
        {"key": "custo_elevado", "icon": "payments", "name": "Custo Hospitalar Elevado",
         "source": "SIH", "est_min": 10, "status": "ok", "linkage": None,
         "note": "Base única SIH. Pipeline completo disponível."},
        {"key": "mortalidade_hospitalar", "icon": "monitor_heart", "name": "Mortalidade Hospitalar",
         "source": "SIH + SIM", "est_min": 15, "status": "ok",
         "linkage": "Alvo principal = óbito intra-hospitalar (MORTE no SIH). Linkage com SIM enriquece como proxy.",
         "note": "Mortalidade durante a internação (SIH). Pipeline completo disponível."},
    ],
    "SINAN": [
        {"key": "abandono_tb", "icon": "pulmonology", "name": "Abandono de Tratamento TB",
         "source": "SINAN", "est_min": 8, "status": "ok", "linkage": None,
         "note": "Base única SINAN_TB. Pipeline completo disponível."},
        {"key": "abandono_hanseniase", "icon": "stethoscope", "name": "Abandono de Tratamento — Hanseníase",
         "source": "SINAN", "est_min": 5, "status": "ok", "linkage": None,
         "note": "Base única SINAN_HANS. Pipeline completo disponível."},
        {"key": "violencia_autoprovocada", "icon": "psychology", "name": "Risco de Violência Autoprovocada",
         "source": "SINAN", "est_min": 10, "status": "ok", "linkage": None,
         "note": "Base única SINAN_VIOL. Pipeline completo disponível."},
        {"key": "intoxicacao_grave", "icon": "warning", "name": "Desfecho Adverso em Intoxicação Exógena",
         "source": "SINAN", "est_min": 10, "status": "ok", "linkage": None,
         "note": "Base única SINAN_IEXO. Pipeline completo disponível."},
        {"key": "dengue_grave", "icon": "pest_control", "name": "Dengue Grave ou com Sinais de Alarme",
         "source": "SINAN", "est_min": 8, "status": "ok", "linkage": None,
         "note": "Base única SINAN_DENG. Arquivo nacional grande, primeiro download mais lento."},
        {"key": "chikungunya_hospitalizado", "icon": "local_hospital", "name": "Hospitalização por Chikungunya",
         "source": "SINAN", "est_min": 6, "status": "ok", "linkage": None,
         "note": "Base única SINAN_CHIK. Pipeline completo disponível."},
        {"key": "obito_aids", "icon": "medical_information", "name": "Óbito por AIDS",
         "source": "SINAN", "est_min": 5, "status": "ok", "linkage": None,
         "note": "Base única SINAN_AIDS. Pipeline completo disponível."},
        {"key": "sifilis_nao_cura", "icon": "medication", "name": "Não-Cura de Sífilis Adquirida",
         "source": "SINAN", "est_min": 5, "status": "ok", "linkage": None,
         "note": "Base única SINAN_SIFA. Prevalência baixa, considere balanceamento."},
    ],
}

# ── Metodologia por desfecho ───────────────────────────────────────────────────
# pull   = como o dado é puxado e a coorte construída
# target = definição operacional do desfecho (como o alvo é derivado)
METHODOLOGY = {
    "baixo_peso_nascer": {
        "pull": "Download direto da base SINASC (nascidos vivos) do estado e ano escolhidos. Coorte = todos os nascimentos. O campo PESO é removido das preditoras para não vazar o desfecho.",
        "target": "Peso ao nascer < 2500 g (campo PESO do SINASC).",
    },
    "prematuridade": {
        "pull": "Base SINASC do estado/ano. Coorte = todos os nascimentos; a idade gestacional (GESTACAO, categórica) é removida das preditoras.",
        "target": "Idade gestacional abaixo de 37 semanas (faixas de GESTACAO).",
    },
    "apgar_baixo": {
        "pull": "Base SINASC do estado/ano. Coorte = nascidos vivos; APGAR1 e APGAR5 saem das preditoras.",
        "target": "Apgar no 5º minuto < 7 (campo APGAR5), marcador de asfixia perinatal.",
    },
    "cesarea": {
        "pull": "Base SINASC do estado/ano. Coorte = todos os nascidos vivos. PARTO e STCESPARTO (proxy de cesárea planejada) saem das preditoras para não vazar o desfecho.",
        "target": "Parto cesáreo (PARTO = 2, contra 1 = vaginal). Features: maternas (idade, escolaridade, raça, estado civil, paridade), pré-natal (consultas), idade gestacional, tipo de gravidez, apresentação fetal e sexo.",
    },
    "anomalia_congenita": {
        "pull": "Base SINASC do estado/ano. IDANOMAL e CODANOMAL (CID da anomalia) saem das preditoras (anti-leakage).",
        "target": "Presença de anomalia/malformação congênita ao nascer (IDANOMAL = 1). Desfecho raro (~0,9%); priorize recall/AUPRC e balanceamento.",
    },
    "mortalidade_neonatal": {
        "pull": "Baixa SINASC (nascimentos) e SIM (óbitos) do estado/ano. Filtra no SIM os óbitos neonatais (idade em horas/dias ≤ 28) e marca os nascimentos cuja chave aparece entre esses óbitos.",
        "target": "Óbito até 28 dias de vida, pareado por linkage determinístico DTNASC + SEXO + PESO entre SINASC e SIM. Pareamento aproximado (sem identificador direto).",
    },
    "permanencia_prolongada": {
        "pull": "Base SIH-RD (internações) do estado/ano. Tempo de permanência calculado de DT_INTER a DT_SAIDA.",
        "target": "Internação com mais de 15 dias de permanência.",
    },
    "uso_uti": {
        "pull": "Base SIH-RD do estado/ano (mensal e pesada; use estados menores ou max_rows controlado). Todas as colunas de UTI (UTI_MES_TO, VAL_UTI, MARCA_UTI, etc.) saem das preditoras.",
        "target": "A internação utilizou UTI, derivado de UTI_MES_TO > 0 (dias-UTI no mês). Features de admissão: idade, sexo, CID principal (capítulo/bloco), permanência, caráter da internação, valor total, raça e procedimento.",
    },
    "infeccao_hospitalar": {
        "pull": "Base SIH-RD do estado/ano. O campo oficial de infecção hospitalar vem vazio no dado público, então usa-se um proxy.",
        "target": "Proxy: presença de CID de infecção no diagnóstico secundário (DIAGSEC1). Aproximação, subestima a incidência real.",
    },
    "custo_elevado": {
        "pull": "Base SIH-RD do estado/ano. O custo é o valor total da AIH (VAL_TOT).",
        "target": "Custo acima do percentil 90 da própria amostra (decil superior). Por construção, ~10% positivos.",
    },
    "mortalidade_hospitalar": {
        "pull": "Base SIH-RD do estado/ano, opcionalmente enriquecida com óbitos do SIM. O alvo principal é a morte registrada na própria AIH.",
        "target": "Óbito durante a internação (campo MORTE do SIH = 1). O link com SIM entra como reforço quando disponível.",
    },
    "abandono_tb": {
        "pull": "Base SINAN-Tuberculose (nacional, filtrada por UF). Coorte = casos encerrados (situação de encerramento conhecida).",
        "target": "Encerramento por abandono de tratamento (SITUA_ENCE).",
    },
    "abandono_hanseniase": {
        "pull": "Base SINAN-Hanseníase (nacional, filtrada por UF). Coorte = casos encerrados.",
        "target": "Saída por abandono de tratamento (TPALTA_N = 3).",
    },
    "violencia_autoprovocada": {
        "pull": "Base SINAN-Violência (nacional, filtrada por UF). LES_AUTOP, CONS_SUIC e VIOL_AUTO são removidas das preditoras (anti-leakage).",
        "target": "Lesão autoprovocada (LES_AUTOP = 1), com consequência suicida / violência autoprovocada como fallback de anos antigos.",
    },
    "intoxicacao_grave": {
        "pull": "Base SINAN-Intoxicação Exógena (nacional, filtrada por UF). Coorte = casos confirmados com evolução conhecida; EVOLUCAO sai das preditoras.",
        "target": "Desfecho adverso: óbito ou incapacidade permanente na evolução do caso.",
    },
    "dengue_grave": {
        "pull": "Base SINAN-Dengue (nacional, filtrada por UF; arquivo grande, 1º download mais lento). CLASSI_FIN sai das preditoras; sinais de alarme ALRM_* entram como features.",
        "target": "Classificação final de dengue com sinais de alarme (CLASSI_FIN = 8) ou dengue grave (CLASSI_FIN = 11).",
    },
    "chikungunya_hospitalizado": {
        "pull": "Base SINAN-Chikungunya (nacional, filtrada por UF). Coorte = casos confirmados.",
        "target": "Necessidade de hospitalização (HOSPITALIZ = 1).",
    },
    "obito_aids": {
        "pull": "Base SINAN-AIDS adulto (nacional, filtrada por UF). Doenças definidoras de AIDS entram como features; EVOLUCAO sai.",
        "target": "Evolução para óbito por AIDS (EVOLUCAO = 2).",
    },
    "sifilis_nao_cura": {
        "pull": "Base SINAN-Sífilis Adquirida (nacional, filtrada por UF). Coorte = casos confirmados com evolução conhecida.",
        "target": "Evolução diferente de cura (falha terapêutica, abandono ou óbito).",
    },
    "readmissao_30d": {
        "pull": "Base SIH-RD do estado/ano. Como o dado público não traz CNS/CPF, o paciente é identificado por chave probabilística (nascimento + sexo + município de residência + CEP). Self-linkage temporal por merge_asof entre alta e novas internações.",
        "target": "Nova internação do mesmo paciente em até 30 dias após a alta. Internação no mesmo dia da alta é tratada como transferência. Exige amostra grande (≥ 30.000) para taxa representativa.",
    },
}


@st.dialog("Metodologia")
def _methodology_dialog(card: dict):
    """Drawer lateral (direita) com a metodologia de obtenção do dado do desfecho."""
    key = card["key"]
    meth = METHODOLOGY.get(key, {})
    st.markdown(f"#### {card['name']}")
    st.caption(f"Fonte: {card['source']}  ·  ~{card['est_min']} min de download")
    st.markdown("---")

    st.markdown("**Como o dado é puxado**")
    st.write(meth.get("pull", "Metodologia em documentação."))

    st.markdown("**Definição do desfecho**")
    st.write(meth.get("target", "—"))

    if card.get("linkage"):
        st.markdown("**Linkage**")
        st.write(card["linkage"])

    if card.get("note"):
        st.markdown("**Ressalvas**")
        st.write(card["note"])

    # Variáveis preditoras (import tardio para não pesar o load da página)
    try:
        from core.outcomes import OUTCOMES
        oc = OUTCOMES.get(key)
        feats = list(getattr(oc, "suggested_features", []) or []) if oc else []
        if feats:
            st.markdown("**Variáveis preditoras sugeridas**")
            st.caption(", ".join(feats))
    except Exception:
        pass


if "outcome_key" not in st.session_state:
    st.session_state.outcome_key = None

_PIPELINE_KEYS = [
    "raw_data", "cohort", "feature_config", "treatment_config",
    "model_config", "model_results", "calib_results", "upload_df",
    "upload_target", "upload_features", "upload_dict", "X_res",
    "benchmark_name", "benchmark_icon", "benchmark_source", "benchmark_description",
]


def _reset_pipeline():
    for _rk in _PIPELINE_KEYS:
        if _rk in st.session_state:
            st.session_state[_rk] = {} if _rk == "raw_data" else None


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
    background-color: #ffffff !important;
    color: #111827 !important;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.block-container { padding: 3rem 3rem !important; max-width: 1200px !important; }

.ms { font-family: 'Material Symbols Outlined'; font-style: normal; font-weight: normal;
    font-size: 1rem; line-height: 1; vertical-align: middle; display: inline-block; color: #111827; }
.ms-lg { font-size: 1.4rem; margin-right: .25rem; }

h1,h2,h3,h4,h5,h6 { font-family:"Space Grotesk","Inter",sans-serif !important; letter-spacing:-.01em; }
.ds-title {
    font-family: "Space Grotesk", "Inter", sans-serif;
    font-size: 1.6rem; font-weight: 700;
    color: #223886 !important; margin-bottom: .15rem;
    display: flex; align-items: center; gap: .35rem;
}
.ds-sub { font-size: 0.9rem; color: #6b7280 !important; margin-bottom: .75rem; }

.ds-legend {
    display: flex; gap: 1.25rem; align-items: center;
    font-size: .75rem; color: #6b7280; margin-bottom: 2rem;
    padding: .6rem 1rem; border: 1px solid #f3f4f6;
    border-radius: 6px; background: #fafafa; width: fit-content;
}
.ds-legend-item { display: flex; align-items: center; gap: .35rem; }
.dot-ok     { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; flex-shrink:0; }
.dot-dev    { width: 8px; height: 8px; border-radius: 50%; background: #f97316; flex-shrink:0; }
.dot-link   { width: 8px; height: 8px; border-radius: 50%; background: #f59e0b; flex-shrink:0; }
.dot-upload { width: 8px; height: 8px; border-radius: 50%; background: #ef4444; flex-shrink:0; }

.ds-group {
    font-size: .68rem; font-weight: 700;
    color: #9ca3af !important; text-transform: uppercase;
    letter-spacing: .1em; margin: 1.75rem 0 .5rem;
    padding-bottom: .35rem; border-bottom: 1px solid #f3f4f6;
}

.ds-card {
    border: 1px solid #e5e7eb; border-radius: 6px;
    padding: .75rem 1rem; margin-bottom: .5rem;
    background: #ffffff !important;
    min-height: 108px;
    display: flex; flex-direction: column;
    justify-content: space-between; overflow: hidden;
    transition: border-color .12s;
}
.ds-card:hover { border-color: #9ca3af; }
.ds-card.sel { border: 1.5px solid #223886; background: #eef1fb !important; }
.ds-card.dev { border-color: #fed7aa; background: #fffbf5 !important; opacity: .9; }
.ds-card.dev:hover { border-color: #fb923c; }
.ds-card.dev.sel { border: 1.5px solid #f97316; background: #fff7ed !important; }
.ds-card.upload { border-color: #fecaca; background: #fff5f5 !important; opacity: .9; }
.ds-card.upload:hover { border-color: #ef4444; }
.ds-card.upload.sel { border: 1.5px solid #ef4444; background: #fee2e2 !important; }

.ds-card-name {
    font-size: .81rem; color: #111827 !important;
    display: flex; align-items: flex-start;
    gap: .3rem; line-height: 1.3; flex: 1; font-weight: 600;
}
.ds-card-name .ms { flex-shrink: 0; margin-top: .05rem; }

.ds-card-meta {
    display: flex; align-items: center; gap: .5rem;
    flex-wrap: wrap; margin-top: .3rem;
}
.ds-badge-src { font-size: .63rem; color: #9ca3af; font-weight: 500; }
.ds-badge-time { font-size: .63rem; color: #6b7280; font-weight: 500;
    display: flex; align-items: center; gap: .2rem; }
.ds-badge-dev { font-size: .6rem; font-weight: 700; color: #c2410c;
    background: #ffedd5; padding: 1px 6px; border-radius: 4px; letter-spacing: .04em; }
.ds-badge-link { font-size: .6rem; font-weight: 600; color: #92400e;
    background: #fef3c7; padding: 1px 6px; border-radius: 4px; }
.ds-badge-upload { font-size: .6rem; font-weight: 700; color: #b91c1c;
    background: #fee2e2; padding: 1px 6px; border-radius: 4px; letter-spacing: .04em; }
.ds-card-note { font-size: .67rem; color: #9ca3af; margin-top: .25rem;
    line-height: 1.35; font-style: italic; }

div[data-testid="stButton"] { display: flex !important; justify-content: flex-start !important; }
div[data-testid="stButton"] > button {
    background-color: #ffffff !important; color: #111827 !important;
    border: 1px solid #e5e7eb !important; border-radius: 6px !important;
    font-size: .8rem !important; font-weight: 500 !important;
    padding: 5px 16px !important; width: auto !important;
    transition: all .12s !important;
}
div[data-testid="stButton"] > button:hover {
    background-color: #eef1fb !important; border-color: #223886 !important;
}
div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
    background-color: #223886 !important; color: #ffffff !important;
    border-color: #223886 !important; font-weight: 600 !important;
    border-bottom: 3px solid #9ec83b !important;
}
div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {
    background-color: #1a2b66 !important; border-color: #1a2b66 !important;
}

/* ── Drawer de Metodologia: ancora o st.dialog na borda direita ───────────── */
@keyframes ds-slide-in { from { transform: translateX(100%); } to { transform: translateX(0); } }
div[data-testid="stDialog"] div[role="dialog"] {
    position: fixed !important;
    top: 0 !important; right: 0 !important; left: auto !important; bottom: 0 !important;
    height: 100vh !important; max-height: 100vh !important;
    width: 460px !important; max-width: 92vw !important;
    border-radius: 0 !important;
    border-left: 4px solid #9ec83b !important;
    box-shadow: -12px 0 32px -16px rgba(15,23,48,.35) !important;
    animation: ds-slide-in .22s ease !important;
    overflow-y: auto !important;
}
div[data-testid="stDialog"] div[role="dialog"] h4 {
    color: #223886 !important; margin-top: 0 !important;
}
/* botão Metodologia: discreto, tipo link */
div[data-testid="stButton"] > button[kind="tertiary"],
div[data-testid="stButton"] > button[data-testid="baseButton-tertiary"] {
    color: #6b7280 !important; font-size: .72rem !important;
    padding: 2px 4px !important; font-weight: 500 !important;
    border: none !important; background: transparent !important;
}
div[data-testid="stButton"] > button[kind="tertiary"]:hover,
div[data-testid="stButton"] > button[data-testid="baseButton-tertiary"]:hover {
    color: #223886 !important; background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ── Voltar ────────────────────────────────────────────────────────────────────
_back_col, _ = st.columns([1, 9])
with _back_col:
    if st.button("← Voltar", key="datasus_back"):
        st.switch_page("app.py")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="ds-title">'
    '<span class="ms ms-lg">local_hospital</span>'
    'DATASUS'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="ds-sub">Desfechos com pipeline pronto sobre bases brasileiras: SINASC, SIH, SIM e SINAN.</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="ds-legend">'
    '<span class="ds-legend-item"><span class="dot-ok"></span>Pipeline completo disponível</span>'
    '<span class="ds-legend-item"><span class="dot-dev"></span>Em desenvolvimento (requer linkage)</span>'
    '<span class="ds-legend-item"><span class="dot-link"></span>Linkage entre bases</span>'
    '<span class="ds-legend-item"><span class="dot-upload"></span>Upload manual necessário</span>'
    '<span style="color:#d1d5db">|</span>'
    '<span class="ds-legend-item"><span class="ms" style="font-size:.85rem;color:#6b7280">schedule</span>'
    'Tempo estimado de download + processamento (1.000 registros)</span>'
    '</div>',
    unsafe_allow_html=True,
)


# ── Grid de outcomes ──────────────────────────────────────────────────────────
try:
    sel = st.session_state.outcome_key
    N_COLS = 4

    for group_name, outcomes in OUTCOME_GROUPS.items():
        _status_order = {"ok": 0, "dev": 1, "upload": 2}
        _sorted = sorted(outcomes, key=lambda o: _status_order.get(o["status"], 1))
        st.markdown(f'<p class="ds-group">{group_name}</p>', unsafe_allow_html=True)

        for row_start in range(0, len(_sorted), N_COLS):
            row = _sorted[row_start : row_start + N_COLS]
            cols = st.columns(N_COLS)

            for col_idx, o in enumerate(row):
                key    = o["key"]
                icon   = o["icon"]
                name   = o["name"]
                source = o["source"]
                status = o["status"]
                est    = o["est_min"]
                linkage = o["linkage"]
                note   = o["note"]
                is_sel = sel == key

                dev_badge    = '<span class="ds-badge-dev">EM DESENVOLVIMENTO</span>' if status == "dev" else ""
                link_badge   = '<span class="ds-badge-link">LINKAGE</span>' if linkage else ""
                upload_badge = '<span class="ds-badge-upload">UPLOAD NECESSÁRIO</span>' if status == "upload" else ""

                card_cls = "ds-card"
                if status == "dev":
                    card_cls += " dev"
                if status == "upload":
                    card_cls += " upload"
                if is_sel:
                    card_cls += " sel"

                note_html = f'<div class="ds-card-note">{note}</div>' if note else ""
                if linkage:
                    note_html += f'<div class="ds-card-note" style="color:#b45309;display:flex;align-items:center;gap:4px"><span class="ms" style="font-size:.8rem">warning</span> {linkage}</div>'

                with cols[col_idx]:
                    st.markdown(
                        f'<div class="{card_cls}">'
                        f'<div class="ds-card-name"><span class="ms">{icon}</span>{name}</div>'
                        f'<div class="ds-card-meta">'
                        f'<span class="ds-badge-src">{source}</span>'
                        f'<span class="ds-badge-time">'
                        f'<span class="ms" style="font-size:.7rem;color:#9ca3af">schedule</span>'
                        f'~{est} min</span>'
                        f'{dev_badge}{upload_badge}{link_badge}'
                        f'</div>'
                        f'{note_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    if status == "ok":
                        btn_label = "Selecionar"
                    elif status == "upload":
                        btn_label = "Aguardando Upload"
                    else:
                        btn_label = "Ver (em dev.)"
                    try:
                        clicked = st.button(
                            btn_label, key=f"sel_{key}",
                            type="primary" if is_sel else "secondary",
                            use_container_width=True,
                        )
                    except TypeError:
                        clicked = st.button(
                            btn_label, key=f"sel_{key}",
                            type="primary" if is_sel else "secondary",
                        )
                    if clicked:
                        if status == "ok":
                            st.session_state.outcome_key = key
                            st.switch_page("pages/analise.py")
                        elif status == "upload":
                            st.session_state.outcome_key = key
                            _reset_pipeline()
                            st.switch_page("pages/upload.py")
                        else:
                            st.toast("Módulo em desenvolvimento — disponível em breve.")

                    try:
                        _meth_clicked = st.button(
                            "ⓘ Metodologia", key=f"meth_{key}",
                            type="tertiary", use_container_width=True,
                        )
                    except TypeError:
                        _meth_clicked = st.button(
                            "ⓘ Metodologia", key=f"meth_{key}",
                            type="secondary",
                        )
                    if _meth_clicked:
                        _methodology_dialog(o)
except Exception as e:
    import traceback
    st.error(f"**Erro na aplicação:** {e}")
    st.code(traceback.format_exc())
