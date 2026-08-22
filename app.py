from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent

st.set_page_config(page_title="Banco Peru - Riesgo Crediticio", layout="wide")
st.title("Banco Peru - Dashboard de Riesgo Crediticio")
st.caption("Modelo de scoring, segmentacion de clientes y control de fairness")

@st.cache_data
def load_data():
    data = pd.read_csv(BASE / "dashboard_data.csv")
    models = pd.read_csv(BASE / "comparacion_modelos.csv", index_col=0).reset_index().rename(columns={"index": "modelo"})
    clusters = pd.read_csv(BASE / "perfil_clusters.csv")
    fairness = pd.read_csv(BASE / "fairness_departamento.csv")
    return data, models, clusters, fairness


df, models, cluster_profile, fairness = load_data()

with st.sidebar:
    st.header("Filtros")
    dept_options = sorted(df["departamento"].dropna().unique())
    channel_options = sorted(df["uso_canal_digital"].dropna().unique())
    risk_options = sorted(df["nivel_riesgo"].dropna().unique())
    cluster_options = sorted(df["cluster"].dropna().unique())

    departments = st.multiselect("Departamento", dept_options, default=dept_options)
    channels = st.multiselect("Uso canal digital", channel_options, default=channel_options)
    risks = st.multiselect("Nivel de riesgo", risk_options, default=risk_options)
    clusters = st.multiselect("Cluster", cluster_options, default=cluster_options)

filtered = df[
    df["departamento"].isin(departments)
    & df["uso_canal_digital"].isin(channels)
    & df["nivel_riesgo"].isin(risks)
    & df["cluster"].isin(clusters)
].copy()

if filtered.empty:
    st.warning("No hay datos para la combinacion de filtros seleccionada.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes", f"{len(filtered):,}")
c2.metric("Riesgo promedio", f"{filtered['prob_default'].mean():.1%}")
c3.metric("Tasa default observada", f"{filtered['target_default'].mean():.1%}")
c4.metric("Aprobacion estimada", f"{(filtered['decision_crediticia'] == 'Aprobar').mean():.1%}")

st.subheader("1. Riesgo crediticio")
trend = (
    filtered.assign(score_decil=pd.qcut(filtered["score_infocorp"], 10, duplicates="drop"))
    .groupby("score_decil", observed=True)
    .agg(score_promedio=("score_infocorp", "mean"), riesgo_promedio=("prob_default", "mean"))
    .reset_index(drop=True)
    .sort_values("score_promedio")
)
fig_line = px.line(
    trend,
    x="score_promedio",
    y="riesgo_promedio",
    markers=True,
    labels={"score_promedio": "Score Infocorp promedio", "riesgo_promedio": "Probabilidad de default"},
    title="Tendencia del riesgo segun score crediticio",
)
st.plotly_chart(fig_line, use_container_width=True)

st.subheader("2. Desempeno de modelos")
models_long = models.melt(
    id_vars=["modelo"],
    value_vars=["roc_auc", "average_precision"],
    var_name="metrica",
    value_name="valor",
)
fig_models = px.bar(
    models_long,
    x="modelo",
    y="valor",
    barmode="group",
    color="metrica",
    range_y=[0, 1],
    labels={"modelo": "Modelo", "valor": "Valor", "metrica": "Metrica"},
    title="ROC-AUC y Average Precision",
)
st.plotly_chart(fig_models, use_container_width=True)

st.subheader("3. Segmentacion de clientes")
col_a, col_b = st.columns(2)
cluster_counts = filtered.groupby("cluster").size().reset_index(name="clientes")
fig_clusters = px.bar(
    cluster_counts,
    x="cluster",
    y="clientes",
    labels={"cluster": "Cluster", "clientes": "Clientes"},
    title="Volumetria de clusters",
)
col_a.plotly_chart(fig_clusters, use_container_width=True)

sample = filtered.sample(min(3000, len(filtered)), random_state=42)
fig_scatter = px.scatter(
    sample,
    x="ingreso_mensual",
    y="saldo_tarjeta",
    color=sample["cluster"].astype(str),
    hover_data=["id_cliente", "departamento", "score_infocorp", "prob_default"],
    labels={"ingreso_mensual": "Ingreso mensual (S/)", "saldo_tarjeta": "Saldo tarjeta (S/)", "color": "Cluster"},
    title="Ingreso vs saldo de tarjeta",
)
col_b.plotly_chart(fig_scatter, use_container_width=True)

st.dataframe(cluster_profile, use_container_width=True, hide_index=True)

st.subheader("4. Fairness regional")
fair_long = fairness.melt(
    id_vars=["departamento"],
    value_vars=["tasa_aprobacion", "fpr"],
    var_name="metrica",
    value_name="valor",
)
fig_fair = px.bar( 
    fair_long,
    x="departamento",
    y="valor",
    color="metrica",
    barmode="group",
    labels={"departamento": "Departamento", "valor": "Tasa", "metrica": "Metrica"},
    title="Tasa de aprobacion y FPR por departamento",
)
st.plotly_chart(fig_fair, use_container_width=True)

st.subheader("5. Exploracion de clientes")
st.dataframe(
    filtered[
        [
            "id_cliente", "departamento", "ingreso_mensual", "saldo_tarjeta",
            "score_infocorp", "edad_cliente", "antiguedad_bancaria_meses",
            "uso_canal_digital", "cluster", "prob_default", "decision_crediticia"
        ]
    ].sort_values("prob_default", ascending=False),
    use_container_width=True,
    hide_index=True,
)
