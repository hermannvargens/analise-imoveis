import streamlit as st
import pandas as pd
import numpy as np
import io
import asyncio
import subprocess
import sys
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
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

# ─── Inicialização do Estado ──────────────────────────────────────────────────
st.set_page_config(page_title="FipeZap Analytics", page_icon="🏠", layout="wide")

if 'df_fipe' not in st.session_state:
    st.session_state.df_fipe = None
if 'pinned_model' not in st.session_state:
    st.session_state.pinned_model = None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏠 Menu")
    app_mode = st.radio("Selecione uma etapa:", ["Extração", "Análise Exploratória (EDA)", "Modelagem"])
    st.markdown("---")
    st.info("Foco: Curitiba | Venda | Número Índice")

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
    st.markdown("## 🤖 Ambiente de Experimentação SARIMA")
    
    if st.session_state.df_fipe is not None:
        df_raw = st.session_state.df_fipe.copy()
        df_raw.set_index('data', inplace=True)
        
        # --- Painel Lateral de Configurações ---
        with st.sidebar:
            st.markdown("### ⚙️ Configurações da Série")
            
            # Filtro de Data
            min_d, max_d = df_raw.index.min().date(), df_raw.index.max().date()
            start_date, end_date = st.date_input("Período de Análise", [min_d, max_d], min_value=min_d, max_value=max_d)
            
            # Tratamento de Outliers
            suavizar = st.checkbox("Suavizar Picos (Média Móvel 3m)")
            
            st.markdown("### 🪟 Divisão Treino/Teste")
            test_size = st.slider("Tamanho do Teste (meses)", 1, 36, 12)
            
            st.markdown("### 🎛️ Parâmetros SARIMA")
            with st.form("parametros_manuais"):
                c1, c2, c3 = st.columns(3)
                p = c1.number_input("p (AR)", 0, 5, 1)
                d = c2.number_input("d (I)", 0, 2, 1)
                q = c3.number_input("q (MA)", 0, 5, 1)
                
                c4, c5, c6 = st.columns(3)
                P = c4.number_input("P (Seasonal)", 0, 3, 0)
                D = c5.number_input("D (Seasonal)", 0, 2, 0)
                Q = c6.number_input("Q (Seasonal)", 0, 3, 0)
                S = st.number_input("S (Períodos/Ciclo)", 0, 24, 12)
                
                btn_manual = st.form_submit_button("Aplicar Parâmetros Manuais")
            
            st.markdown("---")
            btn_auto = st.button("🚀 Auto-Fit (pmdarima)", use_container_width=True)

        # --- Processamento dos Dados ---
        # Aplica filtro de data
        df = df_raw.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)].copy()
        
        # Aplica Suavização
        if suavizar:
            df['indice'] = df['indice'].rolling(window=3, center=True).mean().bfill().ffill()

        train = df['indice'].iloc[:-test_size]
        test = df['indice'].iloc[-test_size:]

        # --- Lógica de Execução ---
        executar_modelo = False
        order = (p, d, q)
        seasonal_order = (P, D, Q, S)

        if btn_auto:
            with st.spinner("Buscando a melhor combinação... (Pode levar alguns segundos)"):
                modelo_auto = auto_arima(train, seasonal=True, m=12, stepwise=True, suppress_warnings=True)
                order = modelo_auto.order
                seasonal_order = modelo_auto.seasonal_order
                st.session_state.last_order = order
                st.session_state.last_seasonal = seasonal_order
                executar_modelo = True
                st.toast(f"Melhor modelo encontrado: {order}x{seasonal_order}")

        if btn_manual or executar_modelo or ('last_order' in st.session_state):
            if not executar_modelo and 'last_order' in st.session_state:
                if not btn_manual: # Usa o estado salvo apenas se não clicou em nada
                    order = st.session_state.last_order
                    seasonal_order = st.session_state.last_seasonal
            
            with st.spinner("Treinando modelo..."):
                try:
                    modelo = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False).fit(disp=False)
                    pred_obj = modelo.get_forecast(steps=test_size)
                    pred_df = pred_obj.summary_frame()
                    
                    st.session_state.last_order = order
                    st.session_state.last_seasonal = seasonal_order
                    
                    # Salva previsões atuais para fixação
                    current_preds = {
                        'mean': pred_df['mean'],
                        'lower': pred_df['mean_ci_lower'],
                        'upper': pred_df['mean_ci_upper'],
                        'label': f"SARIMA {order}x{seasonal_order}"
                    }
                    
                    # --- Métricas em Tempo Real ---
                    mape = mean_absolute_percentage_error(test, pred_df['mean'])
                    rmse = np.sqrt(mean_squared_error(test, pred_df['mean']))
                    mae = mean_absolute_error(test, pred_df['mean'])
                    
                    st.markdown("### 📊 Dashboard de Performance")
                    st.caption(f"Configuração Atual: SARIMA {order} x {seasonal_order}")
                    
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("RMSE", f"{rmse:.2f}")
                    m2.metric("MAE", f"{mae:.2f}")
                    m3.metric("MAPE", f"{mape*100:.2f}%")
                    m4.metric("AIC", f"{modelo.aic:.1f}")
                    m5.metric("BIC", f"{modelo.bic:.1f}")
                    
                    # --- Ação: Fixar Modelo ---
                    if st.button("📌 Fixar Modelo Atual para Comparação"):
                        st.session_state.pinned_model = current_preds
                        st.success("Modelo fixado! Altere os parâmetros para comparar.")

                    # --- Visualização Principal (Plotly) ---
                    st.markdown("---")
                    st.markdown("### 📈 Real vs Predição")
                    
                    fig = go.Figure()
                    
                    # Linha de Treino
                    fig.add_trace(go.Scatter(x=train.index[-24:], y=train.values[-24:], mode='lines', name='Treino (Últimos 24m)', line=dict(color='gray')))
                    
                    # Linha Real do Teste
                    fig.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines+markers', name='Real (Teste)', line=dict(color='blue', width=3)))
                    
                    # Se houver um modelo fixado, plota ele antes
                    if st.session_state.pinned_model:
                        pinned = st.session_state.pinned_model
                        fig.add_trace(go.Scatter(x=test.index, y=pinned['mean'], mode='lines', name=f"Fixado: {pinned['label']}", line=dict(color='orange', dash='dot')))
                    
                    # Linha do Modelo Atual
                    fig.add_trace(go.Scatter(x=test.index, y=pred_df['mean'], mode='lines', name=f"Atual: {current_preds['label']}", line=dict(color='red')))
                    
                    # Intervalo de Confiança do Modelo Atual
                    fig.add_trace(go.Scatter(x=test.index.tolist() + test.index[::-1].tolist(),
                                             y=pred_df['mean_ci_upper'].tolist() + pred_df['mean_ci_lower'][::-1].tolist(),
                                             fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
                                             hoverinfo="skip", showlegend=False, name='Intervalo 95%'))
                    
                    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Gráficos de Diagnóstico Interativos ---
                    st.markdown("---")
                    st.markdown("### 🔬 Diagnóstico de Resíduos")
                    
                    residuos = modelo.resid[1:] # Descarta o primeiro resíduo
                    
                    col_hist, col_corr = st.columns([1, 1])
                    
                    with col_hist:
                        fig_hist = go.Figure(data=[go.Histogram(x=residuos, nbinsx=20, marker_color='#3498db')])
                        fig_hist.update_layout(title="Distribuição dos Resíduos (Normalidade)", height=300, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                    with col_corr:
                        # Calculando ACF e PACF
                        acf_vals = acf(residuos, nlags=20)
                        pacf_vals = pacf(residuos, nlags=20)
                        
                        fig_corr = make_subplots(rows=2, cols=1, subplot_titles=("Autocorrelação (ACF)", "Autocorrelação Parcial (PACF)"), vertical_spacing=0.15)
                        
                        fig_corr.add_trace(go.Bar(x=np.arange(len(acf_vals)), y=acf_vals, marker_color='red', name='ACF'), row=1, col=1)
                        fig_corr.add_trace(go.Bar(x=np.arange(len(pacf_vals)), y=pacf_vals, marker_color='green', name='PACF'), row=2, col=1)
                        
                        # Linhas de significância (Aproximadas para 95% de confiança)
                        conf_level = 1.96 / np.sqrt(len(residuos))
                        fig_corr.add_hline(y=conf_level, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)
                        fig_corr.add_hline(y=-conf_level, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)
                        fig_corr.add_hline(y=conf_level, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
                        fig_corr.add_hline(y=-conf_level, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)

                        fig_corr.update_layout(height=400, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig_corr, use_container_width=True)

                except Exception as e:
                    st.error(f"Erro ao ajustar o modelo com os parâmetros selecionados: {e}")
        else:
            st.info("👆 Selecione os parâmetros manuais ou clique em Auto-Fit para iniciar a modelagem.")
    else:
        st.error("Realize a Extração antes de acessar a modelagem.")
