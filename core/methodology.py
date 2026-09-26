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
        "caveat": "Em desenvolvimento: o linkage por DTNASC + SEXO + PESO deixa chaves nulas casarem entre si e não impõe pareamento 1:1, o que infla os óbitos. Prevalência baixa; considere balanceamento.",
    },
    "permanencia_prolongada": {
        "pull": "Base SIH-RD (internações) do estado/ano. Tempo de permanência calculado de DT_INTER a DT_SAIDA. Uso de UTI sai das preditoras: só existe na alta.",
        "target": "Internação com mais de 15 dias de permanência.",
    },
    "uso_uti": {
        "pull": "Base SIH-RD do estado/ano (mensal e pesada; use estados menores ou max_rows controlado). Todas as colunas de UTI (UTI_MES_TO, VAL_UTI, MARCA_UTI, etc.) E o VAL_TOT saem das preditoras: o valor total da AIH soma o custo de UTI, então seria vazamento.",
        "target": "A internação utilizou UTI, derivado de UTI_MES_TO > 0 (dias-UTI no mês). Features: idade, sexo, CID principal (capítulo/bloco), caráter da internação, raça e procedimento realizado. A permanência sai das preditoras: só existe na alta.",
        "caveat": "VAL_TOT foi excluído por conter o custo de UTI (vazamento). O procedimento (PROC_REA) é o preditor dominante.",
    },
    "infeccao_hospitalar": {
        "pull": "Base SIH-RD do estado/ano. O campo oficial de infecção hospitalar vem vazio no dado público, então usa-se um proxy.",
        "target": "Proxy: presença de CID de infecção no diagnóstico secundário (DIAGSEC1).",
        "caveat": "Por ser proxy, subestima a incidência real de infecção hospitalar.",
    },
    "custo_elevado": {
        "pull": "Base SIH-RD do estado/ano. O custo é o valor total da AIH (VAL_TOT). Permanência e uso de UTI saem das preditoras: só existem na alta.",
        "target": "Custo acima do percentil 90 da própria amostra (decil superior).",
        "caveat": "Por construção (decil superior), a prevalência é ~10%.",
    },
    "mortalidade_hospitalar": {
        "pull": "Base SIH-RD do estado/ano. O alvo é a morte registrada na própria AIH. Permanência, uso de UTI, diárias e valor total saem das preditoras, porque só existem na alta.",
        "target": "Óbito intra-hospitalar (campo MORTE do SIH = 1).",
        "linkage": "O alvo vem só do campo MORTE da AIH. O linkage com o SIM por quase-identificadores não pareia no dado público e não altera o alvo.",
    },
    "readmissao_30d": {
        "pull": "Base SIH-RD do estado/ano. Self-linkage temporal entre alta e novas internações do mesmo paciente, com merge_asof (vetorizado).",
        "target": "Nova internação do mesmo paciente em até 30 dias após a alta. Internação no mesmo dia da alta é tratada como transferência.",
        "linkage": "Como o dado público não traz CNS/CPF, o paciente é identificado por chave probabilística: nascimento + sexo + município de residência + CEP. Metodologia padrão dos estudos brasileiros de readmissão por AIH.",
        "caveat": "A detecção exige ambas as internações na amostra: use amostra grande (≥ 30.000). Amostras pequenas quebram os pares e subestimam a readmissão.",
    },
    "abandono_tb": {
        "pull": "Base SINAN-Tuberculose (nacional, filtrada por UF). Coorte = casos encerrados com situação de encerramento conhecida. Transferência (5), mudança de diagnóstico (6), TB-DR (7) e mudança de esquema (8) são censura: saem da coorte em vez de virar negativo, e a contagem de excluídos vai para o log.",
        "target": "Encerramento por abandono (SITUA_ENCE = 2) ou abandono primário (SITUA_ENCE = 10). Cura (1), óbito por TB (3), óbito por outras causas (4) e falência (9) são negativos.",
        "caveat": "Mapa de SITUA_ENCE conferido com o dicionário do SINAN-TB usado no PySUS e no sinan-continual-learning, mas ainda não contra um arquivo TUBEBR bruto.",
    },
    "obito_tb": {
        "pull": "Base SINAN-Tuberculose (nacional, filtrada por UF). Coorte = casos encerrados; transferência, mudança de diagnóstico, TB-DR e mudança de esquema (SITUA_ENCE 5 a 8) são censura e saem da coorte. SITUA_ENCE e os flags de abandono/cura saem das preditoras (anti-leakage).",
        "target": "Óbito ao encerramento do caso, por TB (SITUA_ENCE = 3) ou por outras causas (SITUA_ENCE = 4). Cura, abandono, falência e abandono primário são negativos. Features de notificação: forma clínica, baciloscopia, cultura, HIV, supervisão do tratamento e demografia.",
        "caveat": "Óbito por outras causas conta como positivo; tratá-lo como risco competitivo é decisão metodológica ainda em aberto.",
    },
    "abandono_hanseniase": {
        "pull": "Base SINAN-Hanseníase (nacional, filtrada por UF). Coorte = casos com tipo de saída registrado (TPALTA_N 1 a 9). Transferências (2 a 5 e 9), óbito (6) e erro diagnóstico (8) são censura: saem da coorte em vez de virar negativo, e a contagem de excluídos vai para o log.",
        "target": "Saída por abandono de tratamento (TPALTA_N = 7), contra a cura (TPALTA_N = 1).",
        "caveat": "Mapa de TPALTA_N conferido com o dicionário do SINAN-Hanseníase (PySUS 0.15.0), mas ainda não contra um arquivo HANSBR bruto. Óbito é censura por risco competitivo, por decisão do lab.",
    },
    "incapacidade_hanseniase": {
        "pull": "Base SINAN-Hanseníase (nacional, filtrada por UF). AVALIA_N/grau_incapacidade (fonte do alvo) e colunas pós-tratamento (TPALTA_N, doses) saem das preditoras.",
        "target": "Incapacidade física grau 2 (G2D) ao diagnóstico (AVALIA_N = 2), indicador de detecção tardia monitorado pela OMS, contra grau zero e grau I (AVALIA_N = 0 ou 1). Não avaliado (AVALIA_N = 3) e em branco saem da coorte em vez de virar negativo. Features: forma clínica, classificação operacional, modo de detecção, baciloscopia, tempo notificação-diagnóstico e demografia.",
        "caveat": "Desfecho moderadamente desbalanceado; AUC modesta (~0,65) por ser predição de detecção tardia a partir de características de base.",
    },
    "dengue_grave": {
        "pull": "Base SINAN-Dengue (nacional, filtrada por UF; arquivo grande, 1º download mais lento). CLASSI_FIN, sinais de alarme (ALRM_*), sinais de gravidade (GRAV_*) e hospitalização saem das preditoras: definem ou decorrem da classificação final.",
        "target": "Classificação final com sinais de alarme (CLASSI_FIN = 11) ou dengue grave (CLASSI_FIN = 12). Coorte = dengue confirmada (CLASSI_FIN 10, 11 ou 12); descartado (5), inconclusivo (8) e chikungunya (13) saem.",
        "caveat": "Códigos do layout 2014 em diante. Notificações no layout anterior (CLASSI_FIN 1 a 4) ficam fora da coorte.",
    },
    "chikungunya_hospitalizado": {
        "pull": "Base SINAN-Chikungunya (nacional, filtrada por UF). Coorte = casos confirmados. Sinais de alarme (ALRM_*) saem das preditoras, porque são critério de internação.",
        "target": "Necessidade de hospitalização (HOSPITALIZ = 1).",
    },
    "obito_aids": {
        "pull": "Base SINAN-AIDS adulto (nacional, filtrada por UF). Coorte = casos com evolução conhecida. Óbito por outras causas (EVOLUCAO = 3) é censura: sai da coorte em vez de virar negativo, e a contagem vai para o log. Doenças definidoras de AIDS entram como features; EVOLUCAO sai.",
        "target": "Evolução para óbito por AIDS (EVOLUCAO = 2), contra vivo (EVOLUCAO = 1).",
        "caveat": "O mapa de EVOLUCAO do SINAN-AIDS ainda não foi conferido com o dicionário oficial nem com um arquivo AIDABR bruto. Óbito por outras causas é censura por risco competitivo, por decisão do lab.",
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
        "pull": "Base SINAN-Intoxicação Exógena (nacional, filtrada por UF). Coorte = intoxicações confirmadas (CLASSI_FIN = 1) com evolução conhecida; EVOLUCAO sai das preditoras. Óbito por outra causa (EVOLUCAO = 4) e perda de seguimento (5) são censura; ignorado (9) e em branco ficam sem rótulo. Os dois grupos saem da coorte em vez de virar negativo, e a contagem vai para o log.",
        "target": "Desfecho adverso: óbito por intoxicação exógena (EVOLUCAO = 3) ou cura com sequela (EVOLUCAO = 2), contra cura sem sequela (EVOLUCAO = 1).",
        "caveat": "Mapa de EVOLUCAO conferido com o dicionário do SINAN-Intoxicação Exógena (PySUS 0.15.0), mas ainda não contra um arquivo IEXOBR bruto. Óbito por outra causa é censura por risco competitivo, por decisão do lab.",
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
