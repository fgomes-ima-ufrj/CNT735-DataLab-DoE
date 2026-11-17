"""CNT735 DataLab DoE v2.2.0 — Assuntos 11–16 | interface Streamlit.

Instalar dependências e iniciar: streamlit run app.py
Camadas matemáticas e de dados funcionam sem Streamlit.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

from cnt735.dados import origem, BB_IDS_BANCO15, box_behnken, METADADOS
from cnt735.modelos import (ajustar, prever, anova_ortogonal, extremos, otimizar_quadratico,
                            sugestao_ensaio, sensibilidade_13)
from cnt735.desenhos import ordem_aleatoria_em_blocos
from cnt735.interoperabilidade import (importar_respostas, ler_tabela, tabelas_modelo,
                                        manifesto, exportar_csv_zip, exportar_ods)
from cnt735.pedagogia import ROTEIROS

st.set_page_config(page_title="CNT735 · DataLab DoE v2.2.1", page_icon="🧪", layout="wide")
TITULOS = {
    "11": "11 · Modelo fatorial 2²", "12": "12 · Réplicas e superfície de resposta",
    "13": "13 · Planejamento fatorial 3³", "14": "14 · Fatorial misto 3×2×3 (CdS)",
    "15": "15 · Box–Behnken × 3³", "16": "16 · Blocos e confundimento",
}
MODO = {"Explorar": "Gráficos e manipulação guiada", "Entender": "Cálculo e interpretação", "Aprofundar": "Rigor, pressupostos e limites"}
COLOR = {"Arco-íris": "Rainbow", "Viridis": "Viridis", "Cividis": "Cividis"}


def store_key(assunto: str) -> str:
    return "13" if assunto == "15" else assunto


def sessao(assunto: str):
    key = store_key(assunto)
    if f"dados_{key}" not in st.session_state:
        d, meta = origem(key)
        st.session_state[f"dados_{key}"] = d
        st.session_state[f"meta_{key}"] = meta
    return st.session_state[f"dados_{key}"], st.session_state[f"meta_{key}"]


def alerta_proveniencia(meta):
    if meta.origem in {"derivado", "simulado", "importado_nao_verificado"}:
        st.warning(f"**ORIGEM: {meta.origem.upper().replace('_', ' ')}.** {meta.observacao} "
                   "Os resultados são demonstrações numéricas, não comprovação experimental.")
    else:
        st.info(f"**ORIGEM: {meta.origem.upper().replace('_',' ')}.** {meta.fonte}. {meta.observacao}")


def editar(assunto: str) -> pd.DataFrame:
    key = store_key(assunto)
    d, meta = sessao(key)
    alerta_proveniencia(meta)
    st.caption("Somente Y é editável. O desenho e os fatores originais são protegidos. "
               "Edite e observe o recálculo. A sessão 13/15 compartilha exatamente a mesma base.")
    widget = f"editar_base_{key}"
    show = st.data_editor(d, key=widget, hide_index=True, use_container_width=True, num_rows="fixed",
                          disabled=[col for col in d if col != "Y"],
                          column_config={"Y": st.column_config.NumberColumn("Y · resposta", format="%.5f", required=True)})
    # Preservar alterações entre telas, inclusive ao sair do Assunto 13 e entrar no 15.
    if not show.Y.equals(d.Y):
        updated = d.copy(deep=True)
        updated["Y"] = show.Y
        st.session_state[f"dados_{key}"] = updated
    else:
        updated = d
    if st.button("↩ Restaurar base histórica", key=f"reset_{key}"):
        original, meta_original = origem(key)
        st.session_state[f"dados_{key}"] = original
        st.session_state[f"meta_{key}"] = meta_original
        if widget in st.session_state: del st.session_state[widget]
        st.rerun()
    with st.expander("📂 Importar respostas CSV, XLSX ou ODS", expanded=False):
        uploaded = st.file_uploader("Tabela com Y e fatores codificados da referência", type=["csv", "xlsx", "ods"],
                                    key=f"upload_{key}")
        if key == "13":
            st.caption("Assuntos 13/15: **A, S e P são obrigatórios**. O ID isolado não é aceito "
                       "porque Aula 16 e banco da Aula 18 utilizam ordenações diferentes.")
        demonstracao = st.checkbox("Os dados importados são medições independentes, cada uma em unidade experimental identificável",
                                   key=f"declara_{key}")
        if uploaded and st.button("Aplicar importação verificada", key=f"aplicar_{key}"):
            try:
                raw = uploaded.getvalue()
                imported = ler_tabela(uploaded.name, raw)
                upd = importar_respostas(d, uploaded.name, raw, key)
                # Não transformar uma simulação em experimento real por mera substituição de Y.
                valid_real = False
                if demonstracao:
                    if "Unidade_Experimental" not in imported:
                        raise ValueError("Para declarar independência, inclua a coluna 'Unidade_Experimental' no arquivo.")
                    unit = imported.Unidade_Experimental
                    if unit.isna().any() or unit.duplicated().any():
                        raise ValueError("Cada medição independente exige Unidade_Experimental única e identificável.")
                    valid_real = True
                previous = st.session_state[f"meta_{key}"]
                new_meta = replace(previous, origem="medido_declarado" if valid_real else "importado_nao_verificado",
                                   fonte=f"Importado por usuário: {uploaded.name}",
                                   certificado=valid_real, independente=valid_real,
                                   observacao="Independência declarada pelo usuário; não constitui auditoria externa." if valid_real
                                   else "Origem/independência não verificadas: inferência bloqueada.")
                st.session_state[f"dados_{key}"] = upd
                st.session_state[f"meta_{key}"] = new_meta
                if widget in st.session_state: del st.session_state[widget]
                st.rerun()
            except (ValueError, KeyError, TypeError) as e:
                st.error(f"Importação recusada: {e}")
    if not np.isfinite(pd.to_numeric(updated.Y, errors="coerce").to_numpy(dtype=float)).all():
        st.error("Respostas com NaN/Inf ou texto inválido: corrija Y.")
        st.stop()
    return updated


def quadro(nome, data):
    st.markdown(f"**{nome}**")
    st.dataframe(data, use_container_width=True, hide_index=False)


def metricas(m):
    cols = st.columns(6)
    for col, title, value in zip(cols,
                                 ("N", "Parâmetros", "gl resíduos", "Posto X", "R² treino", "RMSE treino"),
                                 (str(m["n"]), str(m["p"]), str(m["gl"]), str(m["posto"]),
                                  f"{m['r2']:.5f}", f"{m['rmse']:.5f}")):
        col.metric(title, value)
    st.caption(f"R² ajustado: {fmt(m['r2aj'])} · RSE formal: {fmt(m['rse'])} · "
               f"número de condição X: {m['num_condicao']:.3f} · RMSE LOOCV (descritivo): {fmt(m['rmse_loo'])}")
    if m["inferencial"]:
        st.info("**Inferência condicional habilitada com base em declaração da origem**. " + m["status_inferencia"])
    else:
        st.warning("**Sem inferência experimental validada.** " + m["status_inferencia"])
    if m["tipo"] == "13" and not m["inferencial"]:
        st.caption("A tabela IC_Nominal_Didatico, disponível no modo Aprofundar e na exportação, "
                   "é puramente matemática; não constitui IC experimentalmente certificado.")
    st.caption(f"Erro puro validado: gl={m['gl_erro_puro']} (gl de repetições algébricas={m['gl_erro_puro_algebrico']}). "
               "Falta de ajuste: testável somente com réplicas independentes autênticas.")


def fmt(v):
    return f"{float(v):.5f}" if np.isfinite(v) else "N/D"


def passos(m):
    with st.expander("🧮 Álgebra matricial, coeficientes e verificação", expanded=modo == "Aprofundar"):
        st.latex(r"\hat{\beta}=(X^TX)^{-1}X^Ty\quad\text{(demonstração; estimador numérico por mínimos quadrados)}")
        for nome, frame in (("Matriz X", m["X"]), ("Transposta Xᵀ", m["Xt"]),
                            ("XᵀX", m["XtX"]), ("(XᵀX)⁻¹", m["XtX_inv"]),
                            ("Xᵀy", m["Xty"].to_frame()), ("Coeficientes", m["coef"])):
            quadro(nome, frame)
        if modo == "Aprofundar" and not m["inferencial"] and m["gl"] > 0:
            st.warning("EP e IC nominais abaixo são EXEMPLOS DE ÁLGEBRA sob hipóteses não certificadas. "
                       "Não são evidências experimentais.")
            quadro("EP e IC hipotéticos (NÃO inferenciais)", m["coef_nominal"])


def graficos_basicos(m, ordem=True):
    d = m["pred"]
    xlim = [float(min(d.Y.min(), d.Predito.min())), float(max(d.Y.max(), d.Predito.max()))]
    a, b = st.columns(2)
    with a:
        fig = px.scatter(d, x="Y", y="Predito", hover_data=["Residuo"], title="Observado × predito")
        fig.add_trace(go.Scatter(x=xlim, y=xlim, mode="lines", name="Identidade"))
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(px.histogram(d, x="Residuo", title="Distribuição dos resíduos"), use_container_width=True)
    with b:
        fig = px.scatter(d, x="Predito", y="Residuo", title="Resíduos × ajustados")
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
        if m["gl"] > 0 and len(d) >= 3:
            quantis = stats.norm.ppf((np.arange(1, len(d) + 1) - .5) / len(d))
            qq = pd.DataFrame({"Quantil normal teórico": quantis,
                               "Resíduo ordenado": np.sort(d.Residuo.to_numpy())})
            st.plotly_chart(px.scatter(qq, x="Quantil normal teórico", y="Resíduo ordenado",
                                        title="Q–Q (diagnóstico visual, não valida inferência)"), use_container_width=True)
        else:
            st.info("Modelo saturado: diagnóstico Q-Q não é inferência estatística válida.")
    if ordem:
        order = d["Ordem_Execucao"] if "Ordem_Execucao" in d else np.arange(1, len(d) + 1)
        st.plotly_chart(px.scatter(x=order, y=d.Residuo, labels={"x": "Ordem registrada / linha", "y": "Resíduo"},
                                   title="Resíduos pela ordem disponível (não necessariamente ordem de execução)"),
                        use_container_width=True)
    if modo != "Explorar":
        quadro("Predições, resíduos, alavancagem e Cook", d)
        st.caption("Leverage e Cook são diagnósticos geométricos. Valores altos exigem investigação, não exclusão automática.")


def coef_plot(m):
    z = m["coef"].loc[lambda x: x.Termo != "1"].copy()
    kwargs = {"error_y": "EP"} if m["inferencial"] and z.EP.notna().all() else {}
    st.plotly_chart(px.bar(z, x="Termo", y="Coeficiente", title="Coeficientes (metade do efeito em 2^k ortogonal)", **kwargs),
                    use_container_width=True)


def efeitos(df, factors, pairs):
    cols = st.columns(len(factors))
    for col, factor in zip(cols, factors):
        agg = df.groupby(factor, as_index=False).Y.mean()
        with col:
            st.plotly_chart(px.line(agg, x=factor, y="Y", markers=True, title=f"Médias marginais · {factor}"),
                            use_container_width=True)
    if modo != "Explorar":
        for a, b in pairs:
            agg = df.groupby([a, b], as_index=False).Y.mean()
            agg[b] = agg[b].astype(str)
            st.plotly_chart(px.line(agg, x=a, y="Y", color=b, markers=True, title=f"Interação observada {a} × {b}"),
                            use_container_width=True)


def superficie(m, x, y, fixos=None, comparacao=None, escala_limites=None, passo=41):
    axis = np.linspace(-1, 1, passo)
    gx, gy = np.meshgrid(axis, axis)
    t = pd.DataFrame({x: gx.ravel(), y: gy.ravel()})
    for nome, nivel in (fixos or {}).items(): t[nome] = nivel
    z = (prever(m, t) - prever(comparacao, t) if comparacao is not None else prever(m, t)).reshape(gx.shape)
    if escala_limites:
        vmin, vmax = escala_limites
    elif comparacao is not None:
        mx = max(float(np.max(np.abs(z))), 1e-10)
        vmin, vmax = -mx, mx
    else:
        vmin, vmax = float(z.min()), float(z.max())
    if np.isclose(vmin, vmax): vmax = vmin + 0.01
    st.caption("Malha codificada em [−1,+1] · toda interação é produto dos fatores · "
               "região não observada = previsão condicionada ao modelo.")
    a, b = st.columns(2)
    label = "ΔY" if comparacao else "Y previsto"
    with a:
        fig = go.Figure(go.Surface(x=gx, y=gy, z=z, colorscale=escala, cmin=vmin, cmax=vmax,
                                   colorbar={"title": label}))
        fig.update_layout(title=f"Superfície 3D · {x} × {y}", height=450,
                          scene={"xaxis_title": x, "yaxis_title": y, "zaxis_title": label})
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = go.Figure(go.Contour(x=axis, y=axis, z=z, colorscale=escala, zmin=vmin, zmax=vmax,
                                   contours={"showlabels": True}))
        fig.update_layout(title="Mapa de contorno", xaxis_title=x, yaxis_title=y, height=450)
        st.plotly_chart(fig, use_container_width=True)
    fig = go.Figure(go.Heatmap(x=axis, y=axis, z=z, colorscale=escala, zmin=vmin, zmax=vmax))
    fig.update_layout(title="Mapa de calor", xaxis_title=x, yaxis_title=y, height=420)
    st.plotly_chart(fig, use_container_width=True)
    atmax, atmin = np.unravel_index(np.argmax(z), z.shape), np.unravel_index(np.argmin(z), z.shape)
    st.caption(f"Fatia: máximo da **grade** {z[atmax]:.4f} em {x}={axis[atmax[1]]:.3f}, {y}={axis[atmax[0]]:.3f}; "
               f"mínimo {z[atmin]:.4f}. Não equivale a ótimo comprovado.")


def exportacao(assunto, df, m, *, anova=None, extras=None):
    meta = sessao(assunto)[1]
    tables = tabelas_modelo(df, m, anova=anova, extras=extras)
    info = manifesto(assunto, meta, m)
    with st.expander("📦 Exportar cálculos completos para LibreOffice Calc", expanded=False):
        st.caption("Pacotes incluem dados, X, Xᵀ, XᵀX, inversa, Xᵀy, coeficientes, "
                   "resíduos, ANOVA (quando aplicável), orientações e proveniência.")
        a, b = st.columns(2)
        with a:
            st.download_button("⬇ CSV estruturados + manifesto (ZIP)", exportar_csv_zip(tables, info),
                               file_name=f"CNT735_A{assunto}_Calc_2.2.zip", mime="application/zip",
                               key=f"csv_{assunto}")
        with b:
            try:
                ods = exportar_ods(tables, info)
                st.download_button("⬇ LibreOffice Calc (ODS editável)", ods,
                                   file_name=f"CNT735_A{assunto}_Calc_2.2.ods",
                                   mime="application/vnd.oasis.opendocument.spreadsheet", key=f"ods_{assunto}")
            except ImportError:
                st.warning("ODS depende de odfpy; use requirements.txt ou exporte os CSV.")


def roteiro(assunto):
    r = ROTEIROS[assunto]
    with st.expander("🎓 Prever → experimentar → analisar → interpretar → decidir", expanded=modo == "Explorar"):
        st.markdown(f"**Problema:** {r['pergunta']}")
        st.markdown(f"**Prever antes de calcular:** {r['previsao']}")
        st.text_area("Minha previsão e justificativa", key=f"prev_{assunto}", height=80)
        st.markdown(f"**Observar e explicar:** {r['observacao']}")
        st.markdown(f"**Decidir:** {r['decisao']}")
        st.text_area("Minha interpretação e proposta de próximo experimento", key=f"dec_{assunto}", height=80)
        if modo == "Aprofundar": st.info(r["aprofundamento"])


def assunto11():
    df = editar("11")
    m = ajustar(df, "11", sessao("11")[1])
    metricas(m)
    tabs = st.tabs(["Álgebra matricial", "Efeitos", "Predição e resíduos", "Decisão"])
    with tabs[0]: passos(m); quadro("Vetor β", m["coef"])
    with tabs[1]:
        c = m["coef"].query('Termo != "1"')[["Termo", "Coeficiente"]].copy()
        c["Efeito"] = 2 * c.Coeficiente
        quadro("Efeito = 2β", c); coef_plot(m)
    with tabs[2]: graficos_basicos(m)
    with tabs[3]:
        st.warning("N=p=4: ajuste perfeito é interpolação das quatro observações; "
                   "não há gl residuais, intervalo de confiança ou avaliação externa.")
        roteiro("11")
    exportacao("11", df, m)


def resumo_replica(df, meta):
    rows = []
    # Arithmética DP pode ser didática; EP/IC de pseudo-réplicas são proibidos.
    for (c, t), g in df.groupby(["C", "T"]):
        y = g.Y.to_numpy(dtype=float); n = len(y)
        dp = float(y.std(ddof=1)) if n > 1 else np.nan
        valid = meta.certificado and meta.independente and n >= 2
        ep = dp / np.sqrt(n) if valid else np.nan
        h = stats.t.ppf(.975, n - 1) * ep if valid else np.nan
        rows.append({"C": c, "T": t, "N_linhas": n, "Media": y.mean(), "DP_aritmetico": dp,
                     "EP_inferencial": ep, "IC95_inf": y.mean() - h, "IC95_sup": y.mean() + h,
                     "Margem95": h})
    return pd.DataFrame(rows)


def assunto12():
    df = editar("12"); meta = sessao("12")[1]
    gg = resumo_replica(df, meta)
    quadro("Resumo (DP descritivo não implica variância experimental)", gg)
    gg["Condição"] = gg["C"].astype(str) + " / " + gg["T"].astype(str)
    if gg.Margem95.notna().all():
        st.plotly_chart(px.scatter(gg, x="Condição", y="Media", error_y="Margem95",
                                    title="Média e IC95% condicional"), use_container_width=True)
    else:
        st.plotly_chart(px.scatter(gg, x="Condição", y="Media", title="Médias — IC ocultos por pseudorreplicação"),
                        use_container_width=True)
    m1, m2 = ajustar(df, "12_linear", meta), ajustar(df, "12_curva", meta)
    st.info("Neste desenho C²=T², portanto os coeficientes quadráticos separados NÃO são identificáveis. "
            "Q=(C²+T²)/2 é uma extensão matematicamente estimável que pressupõe divisão igual "
            "da curvatura na superfície; sua forma fora dos pontos medidos NÃO é identificada pelos dados.")
    comp = pd.DataFrame([{"Modelo": k, "R²": m["r2"], "R² ajustado": m["r2aj"],
                          "RMSE": m["rmse"], "Parâmetros": m["p"]}
                         for k, m in (("Primeira ordem + CT", m1), ("Curvatura conjunta Q", m2))])
    quadro("Comparação (treino; não valida predição externa)", comp)
    tabs = st.tabs(["Primeira ordem", "Curvatura", "Superfície", "Redesenho"])
    for tab, m in zip(tabs[:2], (m1, m2)):
        with tab: metricas(m); passos(m); coef_plot(m); graficos_basicos(m)
    with tabs[2]:
        qual = st.radio("Modelo", ["Primeira ordem", "Curvatura conjunta"], horizontal=True)
        superficie(m1 if qual == "Primeira ordem" else m2, "C", "T")
    with tabs[3]:
        st.markdown("**Sugestão:** acrescentar níveis axiais (C=±1,T=0) e (C=0,T=±1) "
                    "com unidades experimentais independentes para separar C² e T².")
        roteiro("12")
    exportacao("12", df, m2, extras={"Resumo_Replicas": gg, "Comparacao": comp})


def assunto13():
    df = editar("13")
    m = ajustar(df, "13", sessao("13")[1])
    metricas(m)
    tabs = st.tabs(["Matriz e IDs", "Diagnósticos", "Efeitos e interações", "Superfícies e extremos"])
    with tabs[0]:
        st.error("**Atenção à numeração:** ID_Banco15 identifica a resposta na planilha Aula 18; "
                 "ID_Aula16 é o ID do desenho sem Y da Aula 16. NUNCA vincule Y por ID entre as fontes.")
        quadro("Mapa completo de reconciliação por tratamentos físicos", df[["ID_Banco15", "ID_Aula16", "Chave_Tratamento", "Acido_pct", "SiC_pct", "Parafina_pct", "Y"]])
        passos(m); quadro("Coeficientes (IC experimental bloqueado)", m["coef"]); coef_plot(m)
    with tabs[1]: graficos_basicos(m)
    with tabs[2]: efeitos(df, ["A", "S", "P"], [("A", "S"), ("A", "P"), ("S", "P")])
    with tabs[3]:
        par = st.selectbox("Par de fatores", [("A", "S"), ("A", "P"), ("S", "P")], format_func=lambda x: " × ".join(x))
        fix = (set(("A", "S", "P")) - set(par)).pop()
        nivel = st.select_slider(f"Fixar {fix}", [-1., 0., 1.], value=0.)
        superficie(m, *par, {fix: nivel})
        with st.expander("Sensibilidade local — gradiente analítico"):
            grd = sensibilidade_13(m, resolucao=9)
            quadro("Regiões com maiores variações previstas (não são automaticamente robustez experimental)",
                   grd.nlargest(12, "Norma_gradiente"))
            st.caption("Norma do gradiente perto de zero implica baixa sensibilidade LOCAL do modelo à mudança codificada. "
                       "Robustez exige ainda incerteza, ruído e validação independente.")
        quadro("Extremos numéricos multistart (não confirmados)", otimizar_quadratico(m, ["A", "S", "P"]))
        roteiro("13")
    exportacao("13", df, m, extras={"Conciliacao_IDs": df[["ID_Banco15", "ID_Aula16", "Chave_Tratamento", "Y"]]})


def assunto14():
    df = editar("14"); m = ajustar(df, "14", sessao("14")[1]); metricas(m)
    tabs = st.tabs(["Modelo", "Diagnósticos", "Efeitos", "Superfícies por C"])
    with tabs[0]:
        st.info("C²=1 em C=±1: não incluir C² além do intercepto. "
                "O modelo atual tem curvatura D² e E² comuns a ambos os níveis de concentração.")
        passos(m); quadro("Coeficientes", m["coef"]); coef_plot(m)
    with tabs[1]: graficos_basicos(m)
    with tabs[2]: efeitos(df, ["D", "C", "E"], [("D", "C"), ("D", "E"), ("C", "E")])
    with tabs[3]:
        # Compartilhar limites para não criar diferença cromática ilusória entre C=-1 e +1.
        a = np.linspace(-1, 1, 41); gx, gy = np.meshgrid(a, a)
        g = pd.DataFrame({"D": gx.ravel(), "E": gy.ravel()})
        vals = []
        for c in (-1., 1.):
            g["C"] = c; vals.append(prever(m, g))
        limites = (float(min(v.min() for v in vals)), float(max(v.max() for v in vals)))
        for c in (-1., 1.):
            st.subheader(f"Concentração C={c:+.0f}")
            superficie(m, "D", "E", {"C": c}, escala_limites=limites)
        st.latex(r"\Delta Y(D,E)=2(\beta_C+\beta_{DC}D+\beta_{CE}E)")
        st.warning("Diferença de **inclinação**, não de curvatura: este modelo não inclui CD², CE² ou CDE. "
                   "Valores iniciais são simulados, não medições de CdS. Segurança: compostos de cádmio são perigosos.")
        roteiro("14")
    exportacao("14", df, m)


def assunto15():
    df = editar("15"); meta = sessao("15")[1]
    sub = box_behnken(df)
    m27, m13 = ajustar(df, "13", meta), ajustar(sub, "13", meta)
    x27 = prever(m27, df); xbb = prever(m13, df)
    comparacao = pd.DataFrame({"ID_Banco15": df.ID_Banco15, "ID_Aula16": df.ID_Aula16,
                                "Chave_Tratamento": df.Chave_Tratamento, "A": df.A, "S": df.S, "P": df.P,
                                "Y": df.Y, "Grupo": np.where(df.ID_Banco15.isin(BB_IDS_BANCO15), "Treino_BB", "Nao_usado_BB"),
                                "Y_pred_27": x27, "Y_pred_BB": xbb,
                                "Erro_BB": df.Y - xbb, "Erro_abs_BB": np.abs(df.Y - xbb),
                                "Delta_BB_menos_27": xbb - x27})
    hold = comparacao[comparacao.Grupo == "Nao_usado_BB"]
    rmse_hold = float(np.sqrt(np.mean(hold.Erro_BB ** 2)))
    resumo = pd.DataFrame([
        {"Plano": "3³ completo", "N": m27["n"], "p": m27["p"], "gl": m27["gl"],
         "R² treino": m27["r2"], "R²aj treino": m27["r2aj"], "RMSE treino": m27["rmse"], "RMSE holdout 14": np.nan},
        {"Plano": "Box–Behnken", "N": m13["n"], "p": m13["p"], "gl": m13["gl"],
         "R² treino": m13["r2"], "R²aj treino": m13["r2aj"], "RMSE treino": m13["rmse"], "RMSE holdout 14": rmse_hold},
    ])
    quadro("Comparação 27 vs 13 — métricas de treino NÃO são diretamente comparáveis", resumo)
    st.metric("RMSE BB nos 14 pontos não utilizados", f"{rmse_hold:.5f}")
    st.warning("A validação é um holdout interno dos 27 registros históricos, não replicação prospectiva. "
               "13/27 ≈ 48% dos ensaios, mas não significa 48% da informação estatística.")
    tabs = st.tabs(["Coeficientes", "Validação dos 14", "Superfícies e ΔY", "Extremos e próximo ensaio"])
    with tabs[0]:
        co = m27["coef"][["Termo", "Coeficiente"]].merge(m13["coef"][["Termo", "Coeficiente"]],
                    on="Termo", suffixes=("_27", "_BB"))
        co["Δcoef"] = co.Coeficiente_BB - co.Coeficiente_27
        quadro("Coeficientes 3³ e Box–Behnken", co)
        st.plotly_chart(px.bar(co, x="Termo", y=["Coeficiente_27", "Coeficiente_BB"], barmode="group"),
                        use_container_width=True)
        quadro("Conjunto BB com IDs físicos e do banco", sub[["ID_Banco15", "ID_Aula16", "Chave_Tratamento", "Y"]])
    with tabs[1]:
        quadro("Erros: 13 pontos ajustados e 14 não utilizados", comparacao)
        a, b = st.columns(2)
        with a:
            st.plotly_chart(px.scatter(comparacao, x="Y", y="Y_pred_BB", color="Grupo",
                                        title="Observado × predito (mesmo modelo BB)"), use_container_width=True)
            st.plotly_chart(px.histogram(comparacao, x="Erro_BB", color="Grupo"), use_container_width=True)
        with b:
            st.plotly_chart(px.bar(comparacao, x="ID_Banco15", y="Erro_BB", color="Grupo"), use_container_width=True)
            st.plotly_chart(px.bar(comparacao, x="ID_Banco15", y="Erro_abs_BB", color="Grupo"), use_container_width=True)
    with tabs[2]:
        par = st.selectbox("Par de fatores", [("A", "S"), ("A", "P"), ("S", "P")], format_func=lambda x: " × ".join(x))
        fix = (set(("A", "S", "P")) - set(par)).pop()
        nivel = st.select_slider(f"Fixar {fix}", [-1., 0., 1.], value=0.)
        # Mesma escala entre os dois modelos para comparação física das cores.
        g = extremos(m27, list(par), fixos={fix: nivel}, resolucao=41)
        valores = np.r_[g.Predito.to_numpy(), prever(m13, g)]
        shared_limits = (float(valores.min()), float(valores.max()))
        st.subheader("Modelo fatorial 27 condições"); superficie(m27, *par, {fix: nivel}, escala_limites=shared_limits)
        st.subheader("Modelo Box–Behnken 13 condições"); superficie(m13, *par, {fix: nivel}, escala_limites=shared_limits)
        st.subheader("ΔY = BB − 3³"); superficie(m13, *par, {fix: nivel}, comparacao=m27)
    with tabs[3]:
        quadro("Extremos numéricos · 27", otimizar_quadratico(m27, ["A", "S", "P"]))
        quadro("Extremos numéricos · BB", otimizar_quadratico(m13, ["A", "S", "P"]))
        proposta = sugestao_ensaio(m13, m27)
        quadro("Dois critérios de próximo ensaio (não equivalentes)", pd.DataFrame([
            {"Critério": k, **{c: r[c] for c in ("A", "S", "P", "Delta_abs", "Ganho_D_condicional")}}
            for k, r in proposta.items()]))
        st.caption("Informação D incremental usa hipótese de erro de variância comum e contraste definido pela matriz do modelo.")
        roteiro("15")
    exportacao("15", df, m27, extras={"Comparacao_13_27": comparacao, "BoxBehnken_13": sub,
                                         "Modelo_BB_Coef": m13["coef"], "Modelo_BB_X": m13["X"],
                                         "Modelo_BB_Pred": m13["pred"], "Modelo_BB_XtX": m13["XtX"],
                                         "Met_Comparacao": resumo})


def cubo(df):
    vert = df.drop_duplicates(["A", "B", "C"]).copy()
    fig = go.Figure()
    records = vert.to_dict(orient="records")
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            if sum(a[f] != b[f] for f in ("A", "B", "C")) == 1:
                fig.add_trace(go.Scatter3d(x=[a["A"], b["A"]], y=[a["B"], b["B"]], z=[a["C"], b["C"]],
                                           mode="lines", line={"color": "gray", "width": 2}, showlegend=False))
    for lote, g in vert.groupby("Lote"):
        fig.add_trace(go.Scatter3d(x=g.A, y=g.B, z=g.C, mode="markers+text", name=lote,
                                   marker={"size": 8}, text=g.Condicao.astype(str)))
    fig.update_layout(title="Cubo 2³ e bloqueamento por ABC",
                      scene={"xaxis_title": "A tempo", "yaxis_title": "B temperatura", "zaxis_title": "C pressão"})
    st.plotly_chart(fig, use_container_width=True)


def assunto16():
    df = editar("16")
    # Bloco e fabricante são dependências do planejamento; mantêm-se bloqueados para edição.
    m = ajustar(df, "16", sessao("16")[1]); metricas(m)
    tabela = anova_ortogonal(df, m)
    tabs = st.tabs(["Contrastes e ANOVA", "Gráficos e diagnósticos", "Cubo e blocos", "Confundimento e redesenho"])
    with tabs[0]:
        passos(m)
        quadro("Contrastes e ANOVA — p oculto em dados simulados", tabela)
        coef_plot(m)
    with tabs[1]:
        st.plotly_chart(px.box(df, x="Lote", y="Y", points="all",
                                title="Força simulada por fabricante: efeito confundido com ABC"), use_container_width=True)
        efeitos(df, ["A", "B", "C"], [("A", "B"), ("A", "C"), ("B", "C")])
        st.plotly_chart(px.bar(tabela.iloc[:7], x="Termo", y="Efeito", title="Efeitos ortogonais e Bloco/ABC"),
                        use_container_width=True)
        graficos_basicos(m)
    with tabs[2]:
        cubo(df)
        quadro("8 tratamentos, 3 simulações por tratamento, 2 blocos",
               df.drop_duplicates("Condicao")[["Condicao", "A", "B", "C", "Bloco", "Lote"]])
        seed = int(st.number_input("Semente para ordem de execução proposta", value=735,
                                   min_value=0, step=1, key="seed_blocos"))
        ordem = ordem_aleatoria_em_blocos(df, seed)
        st.caption("Ordem aleatória proposta DENTRO de cada lote, sem misturar fabricantes. "
                   "Não representa a ordem efetivamente executada.")
        quadro("Sequência planejada para futura coleta", ordem[["Ordem_global_planejada", "Ordem_no_lote_planejada",
             "Lote", "ID", "A", "B", "C", "Bloco", "Seed_ordem"]])
    with tabs[3]:
        st.error("**Bloco=ABC em todas as 24 linhas.** Nem A/B de fabricante nem ABC podem ser estimados "
                 "separadamente, mesmo com p formalmente pequeno.")
        st.markdown("Para dissociar lote e ABC: repetir as oito condições em ambos os lotes "
                    "(16 corridas mínimas; 48 com triplicatas). "
                    "Para generalizar ao fabricante, analisar *múltiplos lotes independentes* por fabricante e efeitos hierárquicos.")
        st.latex(r"\mathrm{rank}[1,A,B,C,AB,AC,BC,Bloco,ABC] = 8 < 9")
        st.warning("Se ABC for fisicamente importante, sua variação será indistinguível da diferença entre lotes.")
        roteiro("16")
    exportacao("16", df, m, anova=tabela,
              extras={"Ordem_Planejada_Blocos": ordem})


st.title("CNT735 · DataLab DoE | Assuntos 11–16")
st.caption("PENt/COPPE/UFRJ · v2.2.1 · rigor científico, reprodutibilidade, investigação orientada e LibreOffice Calc")
st.sidebar.header("Navegação")
selecao = st.sidebar.radio("Assunto", list(TITULOS), format_func=lambda k: TITULOS[k])
modo = st.sidebar.radio("Profundidade pedagógica", list(MODO), index=1,
                        help="As escolhas mudam apenas a apresentação; não alteram a matemática.")
st.sidebar.caption(MODO[modo])
escala = COLOR[st.sidebar.selectbox("Paleta de superfícies", list(COLOR), index=0)]
st.sidebar.caption("Os dados originais são preservados. Edições persistem somente nesta sessão.")
st.sidebar.warning("Dados simulados/derivados ≠ evidência experimental. "
                   "Coeficientes precisos ≠ inferência confiável.")
st.subheader(TITULOS[selecao])
try:
    {"11": assunto11, "12": assunto12, "13": assunto13,
     "14": assunto14, "15": assunto15, "16": assunto16}[selecao]()
except (ValueError, np.linalg.LinAlgError) as err:
    st.error(f"Não é possível analisar estes dados: {err}")
    st.info("Revise os fatores, a finitude das respostas, a codificação, o posto e a origem das observações.")
