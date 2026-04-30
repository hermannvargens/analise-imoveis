import streamlit as st
import pandas as pd
import io
import asyncio
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from playwright.async_api import async_playwright

# ─── Instalação do Chromium ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Preparando ambiente...")
def install_chromium():
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"])
    return True

install_chromium()

# ─── Funções de Scraping e Limpeza ──────────────────────────────────────────
def limpar_dados_fipe(lista_tabelas):
    dfs_processados = []
    meses_map = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
                 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}
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

async def extrair_fipe_curitiba():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(user_agent="Mozilla/5.0")
        try:
            await page.goto("https://www.fipe.org.br/pt-br/indices/fipezap/#indice-mensal", wait_until="networkidle")
            await page.click("#Tipo_chosen")
            await page.locator("#Tipo_chosen .chosen-results li").get_by_text("Venda", exact=True).click()
            await page.wait_for_timeout(500)
            await page.click("#Info_chosen")
            await page.locator("#Info_chosen .chosen-results li").get_by_text("Número Índice", exact=True).click()
            await page.wait_for_timeout(500)
            await page.click("#Regiao_chosen")
            await page.locator("#Regiao_chosen input").fill("Curitiba")
            await page.locator("#Regiao_chosen .chosen-results li").get_by_text("Curitiba", exact=True).click()
            await page.wait_for_timeout(500)
            await page.click("#Dormitorios_chosen")
            await page.locator("#Dormitorios_chosen .chosen-results li").get_by_text("Todos", exact=True).click()
            await page.click("#buttonPesquisar", force=True)
            await page.wait_for_selector("table.results", timeout=30000)
            html = await page.content()
            await browser.close()
            return pd.read_html(io.StringIO(html), attrs={"class": "results"})
        except Exception as e:
            await browser.close()
            raise e

# ─── Configuração de Página e Sidebar ────────────────────────────────────────
st.set_page_config(page_title="FipeZap Analytics", page_icon="🏠", layout="wide")

with st.sidebar:
    st.title("🏠 Menu")
    app_mode = st.radio("Selecione uma etapa:", ["Extração", "Análise Exploratória (EDA)", "Modelagem SARIMA"])
    st.markdown("---")

if 'df_fipe' not in st.session_state:
    st.session_state.df_fipe = None

# ─── ETAPA 1: EXTRAÇÃO ───────────────────────────────────────────────────────
if app_mode == "Extração":
    st.title("🚀 Extração de Dados")
    if st.button("Iniciar Scraping", type="primary"):
        with st.spinner("Extraindo dados da FIPE..."):
            try:
                tabelas = asyncio.run(extrair_fipe_curitiba())
                st.session_state.df_fipe = limpar_dados_fipe(tabelas)
                st.success("Dados extraídos!")
            except Exception as e:
                st.error(f"Erro: {e}")
    
    if st.session_state.df_fipe is not None:
        st.dataframe(st.session_state.df_fipe, use_container_width=True)

# ─── ETAPA 2: EDA ────────────────────────────────────────────────────────────
elif app_mode == "Análise Exploratória (EDA)":
    st.title("📈 Análise Exploratória")
    if st.session_state.df_fipe is not None:
        df = st.session_state.df_fipe.set_index('data')
        st.line_chart(df['indice'])
        
        st.subheader("Decomposição Sazonal")
        dec = seasonal_decompose(df['indice'], model='additive', period=12)
        fig_dec = dec.plot()
        fig_dec.set_size_inches(10, 6)
        st.pyplot(fig_dec)
    else:
        st.warning("Extraia os dados primeiro.")

# ─── ETAPA 3: MODELAGEM SARIMA ────────────────────────────────────────────────
elif app_mode == "Modelagem SARIMA":
    st.title("🤖 Modelagem SARIMA")
    
    if st.session_state.df_fipe is not None:
        df = st.session_state.df_fipe.set_index('data')
        
        # 1. Teste de Estacionaridade
        st.subheader("1. Teste de Estacionaridade (ADF)")
        res_adf = adfuller(df['indice'])
        col1, col2 = st.columns(2)
        col1.metric("Estatística ADF", f"{res_adf[0]:.4f}")
        col2.metric("p-valor", f"{res_adf[1]:.4f}")
        
        if res_adf[1] <= 0.05:
            st.success("Série Estacionária.")
        else:
            st.error("Série não é estacionária (requer diferenciação). O Auto-ARIMA cuidará disso.")

        # 2. Busca do melhor modelo
        st.subheader("2. Busca do Melhor Modelo (Auto-ARIMA)")
        
        @st.cache_data
        def treinar_auto_arima(serie):
            return auto_arima(serie, seasonal=True, m=12, stepwise=True, 
                              suppress_warnings=True, error_action="ignore")

        with st.spinner("Otimizando parâmetros p, d, q..."):
            modelo_auto = treinar_auto_arima(df['indice'])
            st.code(f"Melhor Modelo: {modelo_auto.order}x{modelo_auto.seasonal_order}")

        # 3. Validação Cronológica (Últimos 12 meses)
        st.subheader("3. Validação (Últimos 12 meses)")
        train = df['indice'].iloc[:-12]
        test = df['indice'].iloc[-12:]

        with st.spinner("Validando modelo..."):
            modelo_sarima = SARIMAX(train, order=modelo_auto.order, 
                                    seasonal_order=modelo_auto.seasonal_order).fit(disp=False)
            
            pred_obj = modelo_sarima.get_forecast(steps=12)
            pred_df = pred_obj.summary_frame()
            pred_df.index = test.index

        # Métricas
        mape = mean_absolute_percentage_error(test, pred_df['mean'])
        rmse = np.sqrt(mean_squared_error(test, pred_df['mean']))
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("MAPE", f"{mape*100:.4f}%")
        m_col2.metric("RMSE", f"{rmse:.2f}")

        # Gráfico de Validação
        fig_val, ax_val = plt.subplots(figsize=(10, 4))
        ax_val.plot(train.iloc[-24:], label="Treino (últimos 2 anos)")
        ax_val.plot(test, label="Real (Teste)", color="black", fontweight='bold')
        ax_val.plot(pred_df['mean'], label="Previsto", color="red", linestyle="--")
        ax_val.fill_between(pred_df.index, pred_df['mean_ci_lower'], pred_df['mean_ci_upper'], color='pink', alpha=0.3)
        ax_val.legend()
        st.pyplot(fig_val)

        # 4. Diagnóstico
        st.subheader("4. Diagnóstico de Resíduos")
        fig_diag = modelo_sarima.plot_diagnostics(figsize=(10, 8))
        plt.tight_layout()
        st.pyplot(fig_diag)

        # 5. Previsão Futura
        st.subheader("5. Previsão para os Próximos 12 Meses")
        if st.button("Gerar Previsão Futura"):
            modelo_final = SARIMAX(df['indice'], order=modelo_auto.order, 
                                   seasonal_order=modelo_auto.seasonal_order).fit(disp=False)
            futuro_obj = modelo_final.get_forecast(steps=12)
            futuro_df = futuro_obj.summary_frame()
            
            # Criar datas futuras
            datas_futuras = pd.date_range(start=df.index.max() + pd.DateOffset(months=1), periods=12, freq='MS')
            futuro_df.index = datas_futuras
            
            st.write("Projeção de valores do índice:")
            st.dataframe(futuro_df[['mean', 'mean_ci_lower', 'mean_ci_upper']])
            
            fig_fut, ax_fut = plt.subplots(figsize=(10, 4))
            ax_fut.plot(df['indice'].iloc[-24:], label="Histórico recente")
            ax_fut.plot(futuro_df['mean'], label="Previsão Futura", color="green")
            ax_fut.fill_between(futuro_df.index, futuro_df['mean_ci_lower'], futuro_df['mean_ci_upper'], color='green', alpha=0.1)
            ax_fut.legend()
            st.pyplot(fig_fut)
    else:
        st.error("Extraia os dados primeiro.")
