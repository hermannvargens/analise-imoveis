import streamlit as st
import pandas as pd
import numpy as np
import io
import asyncio
import subprocess
import sys
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from playwright.async_api import async_playwright

# ─── Instalação do Chromium ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Preparando ambiente (Chromium)...")
def install_chromium():
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], capture_output=True)
    return True

install_chromium()

# ─── Funções de Suporte (Scraping e Limpeza) ──────────────────────────────────
def limpar_dados_fipe(lista_tabelas):
    dfs_processados = []
    meses_map = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}
    ano_corrente = 2026
    for df in lista_tabelas:
        if df.empty or df.shape[1] < 2: continue
        temp_df = df.copy().iloc[:, [0, 1]]
        temp_df.columns = ['mes_str', 'valor']
        temp_df['mes_num'] = temp_df['mes_str'].astype(str).str.lower().str.strip().map(meses_map)
        temp_df = temp_df.dropna(subset=['mes_num'])
        if temp_df.empty: continue
        temp_df['Ano'] = ano_corrente
        dfs_processados.append(temp_df)
        ano_corrente -= 1 
    df_final = pd.concat(dfs_processados, ignore_index=True)
    df_final['data'] = pd.to_datetime(df_final['Ano'].astype(str) + '-' + df_final['mes_num'].astype(str) + '-01')
    df_series = df_final[['data', 'valor']].sort_values('data').reset_index(drop=True)
    df_series.columns = ['data', 'indice']
    df_series['indice'] = df_series['indice'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df_series['indice'] = df_series['indice'].str.extract(r'(\d+\.?\d*)').astype(float)
    return df_series

async def extrair_fipe_curitiba(log_container):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
        page = await browser.new_page(user_agent="Mozilla/5.0")
        try:
            await page.goto("https://www.fipe.org.br/pt-br/indices/fipezap/#indice-mensal", wait_until="networkidle")
            await page.click("#Tipo_chosen")
            await page.locator("#Tipo_chosen .chosen-results li").get_by_text("Venda", exact=True).click()
            await page.click("#Info_chosen")
            await page.locator("#Info_chosen .chosen-results li").get_by_text("Número Índice", exact=True).click()
            await page.click("#Regiao_chosen")
            await page.locator("#Regiao_chosen input").fill("Curitiba")
            await page.wait_for_timeout(500)
            await page.locator("#Regiao_chosen .chosen-results li").get_by_text("Curitiba", exact=True).click()
            await page.click("#Dormitorios_chosen")
            await page.locator("#Dormitorios_chosen .chosen-results li").get_by_text("Todos", exact=True).click()
            log_container.info("🔍 Consultando base FIPE...")
            await page.click("#buttonPesquisar", force=True)
            await page.wait_for_selector("table.results", state="visible", timeout=30_000)
            html = await page.content()
            await browser.close()
            return pd.read_html(io.StringIO(html), attrs={"class": "results"})
        except Exception as e:
            await browser.close()
            raise e

# ─── Layout e Sidebar ────────────────────────────────────────────────────────
st.set_page_config(page_title="FipeZap Analytics", page_icon="🏠", layout="wide")

with st.sidebar:
    st.title("🏠 Menu")
    app_mode = st.radio("Selecione uma etapa:", ["Extração", "Análise Exploratória (EDA)", "Modelagem"])
    st.markdown("---")
    st.info("Foco: Curitiba | Venda | Número Índice")

if 'df_fipe' not in st.session_state:
    st.session_state.df_fipe = None

# ─── ETAPA 1: EXTRAÇÃO ───────────────────────────────────────────────────────
if app_mode == "Extração":
    st.markdown("## 📥 Extração de Dados")
    col_btn, _ = st.columns([2, 5])
    with col_btn:
        btn_extract = st.button("🚀 Iniciar Scraping", use_container_width=True, type="primary")
    log_place = st.empty()
    if btn_extract:
        with st.spinner("Extraindo dados da FIPE..."):
            try:
                tabelas = asyncio.run(extrair_fipe_curitiba(log_place))
                st.session_state.df_fipe = limpar_dados_fipe(tabelas)
                st.success("Dados prontos!")
            except Exception as e: st.error(f"Erro: {e}")
    if st.session_state.df_fipe is not None:
        st.dataframe(st.session_state.df_fipe, use_container_width=True)

# ─── ETAPA 2: EDA ────────────────────────────────────────────────────────────
elif app_mode == "Análise Exploratória (EDA)":
    st.markdown("## 📈 Análise Exploratória")
    if st.session_state.df_fipe is not None:
        df = st.session_state.df_fipe.set_index('data')
        st.line_chart(df['indice'])
        dec = seasonal_decompose(df['indice'], model='additive', period=12)
        st.pyplot(dec.plot())
    else: st.warning("Extraia os dados primeiro.")

# ─── ETAPA 3: MODELAGEM (SARIMA) ─────────────────────────────────────────────
elif app_mode == "Modelagem":
    st.markdown("## 🤖 Modelagem SARIMA")
    
    if st.session_state.df_fipe is not None:
        df = st.session_state.df_fipe.copy()
        
        # 1. Teste de Estacionaridade
        st.subheader("1. Teste de Estacionaridade (ADF)")
        res = adfuller(df['indice'])
        col1, col2 = st.columns(2)
        col1.write(f"**Estatística ADF:** {res[0]:.4f}")
        col2.write(f"**p-valor:** {res[1]:.4f}")
        if res[1] <= 0.05:
            st.success("A série é estacionária.")
        else:
            st.warning("A série não é estacionária (precisa de diferenciação). O Auto-ARIMA cuidará disso.")

        st.markdown("---")
        st.subheader("2. Treinamento do Modelo Auto-SARIMA")
        
        if st.button("🛠️ Treinar Melhor Modelo (Auto-ARIMA)"):
            with st.spinner("Buscando hiperparâmetros (p, d, q) x (P, D, Q, S)..."):
                modelo_auto = auto_arima(df['indice'], seasonal=True, m=12, stepwise=True, 
                                       suppress_warnings=True, error_action="ignore")
                
                st.session_state.modelo_auto = modelo_auto
                st.write(f"**Melhor Modelo:** {modelo_auto.order} x {modelo_auto.seasonal_order}")

        if 'modelo_auto' in st.session_state:
            st.markdown("---")
            st.subheader("3. Validação Cronológica (Últimos 12 meses)")
            
            # Split
            train = df['indice'].iloc[:-12]
            test = df['indice'].iloc[-12:]
            
            # Ajuste no treino
            with st.spinner("Validando modelo nos dados de teste..."):
                modelo_val = SARIMAX(train, order=st.session_state.modelo_auto.order, 
                                   seasonal_order=st.session_state.modelo_auto.seasonal_order).fit(disp=False)
                
                pred_obj = modelo_val.get_forecast(steps=12)
                pred_df = pred_obj.summary_frame()
                
                # Métricas
                mape = mean_absolute_percentage_error(test, pred_df['mean'])
                rmse = np.sqrt(mean_squared_error(test, pred_df['mean']))
                
                c1, c2 = st.columns(2)
                c1.metric("MAPE", f"{mape*100:.4f}%")
                c2.metric("RMSE", f"{rmse:.2f} pts")

                # Gráfico de Comparação
                fig_comp, ax = plt.subplots(figsize=(10, 4))
                ax.plot(test.index, test.values, label='Real (Teste)', marker='o')
                ax.plot(test.index, pred_df['mean'], label='Predição SARIMA', color='red', linestyle='--')
                ax.fill_between(test.index, pred_df['mean_ci_lower'], pred_df['mean_ci_upper'], color='pink', alpha=0.3)
                ax.legend()
                ax.set_title("Real vs Predição (Últimos 12 meses)")
                st.pyplot(fig_comp)

                # Diagnósticos
                st.subheader("4. Diagnóstico de Resíduos")
                fig_diag = modelo_val.plot_diagnostics(figsize=(10, 8))
                st.pyplot(fig_diag)
    else:
        st.error("Realize a Extração antes de modelar.")
