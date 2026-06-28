"""Metodologia por desfecho + drawer lateral reutilizável (grid e pipeline).

Cada entrada descreve:
  pull    = como o dado é puxado e a coorte construída
  target  = definição operacional do desfecho (como o alvo é derivado)
  linkage = (opcional) método de pareamento entre registros
  caveat  = (opcional) ressalva metodológica relevante
"""
from __future__ import annotations

# NB: streamlit é importado de forma tardia dentro de show_methodology para que
# o dicionário METHODOLOGY (dado puro) possa ser importado sem streamlit — útil
# para os testes offline de CI.

METHODOLOGY: dict[str, dict] = {
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
        "target": "Presença de anomalia/malformação congênita ao nascer (IDANOMAL = 1).",
        "caveat": "Desfecho raro (~0,9%) e desbalanceado: priorize recall/AUPRC e use balanceamento (class weight ou SMOTE).",
    },
    "mortalidade_neonatal": {
        "pull": "Baixa SINASC (nascimentos) e SIM (óbitos) do estado/ano. Filtra no SIM os óbitos neonatais (idade em horas/dias ≤ 28) e marca os nascimentos cuja chave aparece entre esses óbitos.",
        "target": "Óbito até 28 dias de vida.",
        "linkage": "Linkage determinístico SINASC ↔ SIM pela chave DTNASC + SEXO + PESO. Pareamento por quase-identificadores, já que o dado público não traz identificador direto do recém-nascido.",
        "caveat": "Prevalência baixa; considere balanceamento.",
    },
    "permanencia_prolongada": {
        "pull": "Base SIH-RD (internações) do estado/ano. Tempo de permanência calculado de DT_INTER a DT_SAIDA.",
        "target": "Internação com mais de 15 dias de permanência.",
    },
    "uso_uti": {
        "pull": "Base SIH-RD do estado/ano (mensal e pesada; use estados menores ou max_rows controlado). Todas as colunas de UTI (UTI_MES_TO, VAL_UTI, MARCA_UTI, etc.) E o VAL_TOT saem das preditoras: o valor total da AIH soma o custo de UTI, então seria vazamento.",
        "target": "A internação utilizou UTI, derivado de UTI_MES_TO > 0 (dias-UTI no mês). Features de admissão: idade, sexo, CID principal (capítulo/bloco), permanência, caráter da internação, raça e procedimento realizado.",
        "caveat": "VAL_TOT foi excluído por conter o custo de UTI (vazamento). O procedimento (PROC_REA) é o preditor dominante.",
    },
    "infeccao_hospitalar": {
        "pull": "Base SIH-RD do estado/ano. O campo oficial de infecção hospitalar vem vazio no dado público, então usa-se um proxy.",
        "target": "Proxy: presença de CID de infecção no diagnóstico secundário (DIAGSEC1).",
        "caveat": "Por ser proxy, subestima a incidência real de infecção hospitalar.",
    },
    "custo_elevado": {
        "pull": "Base SIH-RD do estado/ano. O custo é o valor total da AIH (VAL_TOT).",
        "target": "Custo acima do percentil 90 da própria amostra (decil superior).",
        "caveat": "Por construção (decil superior), a prevalência é ~10%.",
    },
    "mortalidade_hospitalar": {
        "pull": "Base SIH-RD do estado/ano. O alvo é a morte registrada na própria AIH.",
        "target": "Óbito durante a internação (campo MORTE do SIH = 1).",
        "linkage": "O óbito intra-hospitalar vem direto do campo MORTE da AIH. O link com o SIM por quase-identificadores entra como reforço quando disponível.",
    },
    "readmissao_30d": {
        "pull": "Base SIH-RD do estado/ano. Self-linkage temporal entre alta e novas internações do mesmo paciente, com merge_asof (vetorizado).",
        "target": "Nova internação do mesmo paciente em até 30 dias após a alta. Internação no mesmo dia da alta é tratada como transferência.",
        "linkage": "Como o dado público não traz CNS/CPF, o paciente é identificado por chave probabilística: nascimento + sexo + município de residência + CEP. Metodologia padrão dos estudos brasileiros de readmissão por AIH.",
        "caveat": "A detecção exige ambas as internações na amostra: use amostra grande (≥ 30.000). Amostras pequenas quebram os pares e subestimam a readmissão.",
    },
    "abandono_tb": {
        "pull": "Base SINAN-Tuberculose (nacional, filtrada por UF). Coorte = casos encerrados (situação de encerramento conhecida).",
        "target": "Encerramento por abandono de tratamento (SITUA_ENCE = abandono).",
    },
    "abandono_hanseniase": {
        "pull": "Base SINAN-Hanseníase (nacional, filtrada por UF). Coorte = casos encerrados.",
        "target": "Saída por abandono de tratamento (TPALTA_N = 3).",
    },
    "dengue_grave": {
        "pull": "Base SINAN-Dengue (nacional, filtrada por UF; arquivo grande, 1º download mais lento). CLASSI_FIN sai das preditoras; sinais de alarme ALRM_* entram como features.",
        "target": "Classificação final com sinais de alarme (CLASSI_FIN = 8) ou dengue grave (CLASSI_FIN = 11).",
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
        "caveat": "Prevalência baixa; considere balanceamento.",
    },
    "violencia_autoprovocada": {
        "pull": "Base SINAN-Violência (nacional, filtrada por UF). LES_AUTOP, CONS_SUIC e VIOL_AUTO são removidas das preditoras (anti-leakage).",
        "target": "Lesão autoprovocada (LES_AUTOP = 1), com consequência suicida / violência autoprovocada como fallback de anos antigos.",
    },
    "intoxicacao_grave": {
        "pull": "Base SINAN-Intoxicação Exógena (nacional, filtrada por UF). Coorte = casos confirmados com evolução conhecida; EVOLUCAO sai das preditoras.",
        "target": "Desfecho adverso: óbito ou incapacidade permanente na evolução do caso.",
    },
}


_DRAWER_CSS = """
<style>
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
div[data-testid="stDialog"] div[role="dialog"] h4 { color: #223886 !important; margin-top: 0 !important; }
</style>
"""


def show_methodology(outcome_key: str):
    """Abre o drawer lateral (direita) com a metodologia do desfecho.

    Reutilizável no grid DATASUS e no pipeline (sidebar). O `st.dialog` é aplicado
    aqui (import tardio) para não exigir streamlit ao importar o módulo.
    """
    import streamlit as st

    @st.dialog("Metodologia")
    def _dialog():
        _render_methodology(outcome_key)

    _dialog()


def _render_methodology(outcome_key: str):
    import streamlit as st

    st.markdown(_DRAWER_CSS, unsafe_allow_html=True)
    meth = METHODOLOGY.get(outcome_key, {})

    name = outcome_key
    source = ""
    est = None
    feats: list[str] = []
    try:
        from core.outcomes import OUTCOMES
        oc = OUTCOMES.get(outcome_key)
        if oc is not None:
            name = oc.name
            source = " + ".join(oc.data_sources)
            est = getattr(oc, "estimated_download_min", None)
            feats = list(getattr(oc, "suggested_features", []) or [])
    except Exception:
        pass

    st.markdown(f"#### {name}")
    cap = f"Fonte: {source}" if source else ""
    if est:
        cap += f"  ·  ~{est} min de download"
    if cap:
        st.caption(cap)
    st.markdown("---")

    if not meth:
        st.write("Metodologia desta base ainda não documentada.")
        return

    st.markdown("**Como o dado é puxado**")
    st.write(meth.get("pull", "—"))

    st.markdown("**Definição do desfecho**")
    st.write(meth.get("target", "—"))

    if meth.get("linkage"):
        st.markdown("**Linkage**")
        st.write(meth["linkage"])

    if meth.get("caveat"):
        st.markdown("**Ressalvas**")
        st.write(meth["caveat"])

    if feats:
        st.markdown("**Variáveis preditoras sugeridas**")
        st.caption(", ".join(feats))


def has_methodology(outcome_key: str) -> bool:
    return outcome_key in METHODOLOGY
