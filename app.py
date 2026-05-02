import streamlit as st
import pandas as pd
import numpy as np
import io
import asyncio
import subprocess
import sys
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from playwright.async_api import async_playwright

# ─── Listas de Seleção ───────────────────────────────────────────────────────
CIDADES = [
    "Aracaju", "Balneário Camboriú", "Barueri", "Belém", "Belo Horizonte", "Betim", 
    "Blumenau", "Campinas", "Campo Grande", "Canoas", "Caxias do Sul", "Contagem", 
    "Cuiabá", "Curitiba", "Diadema", "Distrito Federal", "FipeZap", "Florianópolis", 
    "Fortaleza", "Goiânia", "Guarujá", "Guarulhos", "Itajaí", "Itapema", 
    "Jaboatão dos Guararapes", "João Pessoa", "Joinville", "Londrina", "Maceió", 
    "Manaus", "Natal", "Niterói", "Novo Hamburgo", "Osasco", "Pelotas", 
    "Porto Alegre", "Praia Grande", "Recife", "Ribeirão Preto", "Rio de Janeiro", 
    "Salvador", "Santa Maria", "Santo André", "Santos", "São Bernardo do Campo", 
    "São Caetano do Sul", "São José", "São José do Rio Preto", "São José dos Campos", 
    "São José dos Pinhais", "São Leopoldo", "São Luís", "São Paulo", "São Vicente", 
    "Teresina", "Vila Velha", "Vitória"
]

DORMITORIOS = ["Todos", "1", "2", "3", "4"]

# ─── Instalação do Chromium ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Preparando ambiente (Chromium)...")
def install_chromium():
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], capture_output=True)
    return True

install_chromium()

# ─── Funções de Dados (FIPE e SGS do BCB) ───────────────────────────────────
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

async def extrair_fipe_dinamico(cidade, dormitorios, log_container):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
        page = await browser.new_page(user_agent="Mozilla/5.0")
        try:
            await page.goto("https://www.fipe.org.br/pt-br/indices/fipezap/#indice-mensal", wait_until="networkidle")
            
            log_container.info("🖱️ Configurando Tipo e Informação...")
            await page.click("#Tipo_chosen")
            await page.locator("#Tipo_chosen .chosen-results li").get_by_text("Venda", exact=True).click()
            
            await page.click("#Info_chosen")
            await page.locator("#Info_chosen .chosen-results li").get_by_text("Número Índice", exact=True).click()
            
            log_container.info(f"📍 Selecionando Região: {cidade}...")
            await page.click("#Regiao_chosen")
            await page.locator("#Regiao_chosen input").fill(cidade)
            await page.wait_for_timeout(500)
            await page.locator("#Regiao_chosen .chosen-results li").get_by_text(cidade, exact=True).click()
            
            log_container.info(f"🛏️ Selecionando Dormitórios: {dormitorios}...")
            await page.click("#Dormitorios_chosen")
            await page.locator("#Dormitorios_chosen .chosen-results li").get_by_text(dormitorios, exact=True).click()
            
            log_container.info("🔍 Consultando base FIPE...")
            await page.click("#buttonPesquisar", force=True)
            await page.wait_for_selector("table.results", state="visible", timeout=30_000)
            
            html = await page.content()
            await browser.close()
            return pd.read_html(io.StringIO(html), attrs={"class": "results"})
        except Exception as e:
            await browser.close()
            raise e

@st.cache_data(show_spinner=False)
def buscar_dados_sgs(codigo):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json"
    response = requests.get(url)
    df = pd.read_json(io.StringIO(response.text))
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df['valor'] = pd.to_numeric(df['valor']) / 100 
    return df

# ─── Inicialização do Estado ──────────────────────────────────────────────────
st.set_page_config(page_title="FipeZap Analytics", page_icon="🏠", layout="wide")

if 'df_fipe' not in st.session_state:
    st.session_state.df_fipe = None
if 'pinned_model' not in st.session_state:
    st.session_state.pinned_model = None
if 'cidade_atual' not in st.session_state:
    st.session_state.cidade_atual = "Curitiba"
if 'dormitorios_atual' not in st.session_state:
    st.session_state.dormitorios_atual = "Todos"

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏠 Menu")
    app_mode = st.radio("Selecione uma etapa:", ["Extração", "Análise Exploratória (EDA)", "Análise de Valor Real", "Modelagem"])
    st.markdown("---")
    st.info("Foco Base: Venda | Número Índice")

# Formatação auxiliar para os títulos dinâmicos
titulo_dinamico = f"{st.session_state.cidade_atual} ({st.session_state.dormitorios_atual} dorms)" if st.session_state.dormitorios_atual != "Todos" else f"{st.session_state.cidade_atual} (Todos os dorms)"

# ─── ETAPA 1: EXTRAÇÃO ───────────────────────────────────────────────────────
if app_mode == "Extração":
    st.markdown("## 📥 Extração de Dados FipeZap")
    st.write("Selecione os filtros abaixo para extrair a série histórica.")
    
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            cidade_selecionada = st.selectbox("Selecione a Região/Cidade:", CIDADES, index=CIDADES.index("Curitiba"))
        with col2:
            dormitorios_selecionados = st.selectbox("Quantidade de Dormitórios:", DORMITORIOS, index=0)
            
        btn_extract = st.button("🚀 Extrair Dados Agora", use_container_width=True, type="primary")
    
    log_place = st.empty()
    
    if btn_extract:
        with st.spinner(f"Extraindo dados de {cidade_selecionada}..."):
            try:
                tabelas = asyncio.run(extrair_fipe_dinamico(cidade_selecionada, dormitorios_selecionados, log_place))
                st.session_state.df_fipe = limpar_dados_fipe(tabelas)
                st.session_state.cidade_atual = cidade_selecionada
                st.session_state.dormitorios_atual = dormitorios_selecionados
                # Força a atualização do log para limpar mensagens antigas
                log_place.empty()
                st.success(f"✅ Dados de {cidade_selecionada} ({dormitorios_selecionados}) extraídos com sucesso!")
            except Exception as e: 
                log_place.empty()
                st.error(f"❌ Erro na extração: {e}")
                
    if st.session_state.df_fipe is not None:
        st.markdown(f"### Visualização dos Dados: {titulo_dinamico}")
        st.dataframe(st.session_state.df_fipe, use_container_width=True)

# ─── ETAPA 2: EDA ────────────────────────────────────────────────────────────
elif app_mode == "Análise Exploratória (EDA)":
    st.markdown(f"## 📈 Análise Exploratória: {titulo_dinamico}")
    if st.session_state.df_fipe is not None:
        df = st.session_state.df_fipe.set_index('data')
        st.line_chart(df['indice'])
        dec = seasonal_decompose(df['indice'], model='additive', period=12)
        st.pyplot(dec.plot())
    else: st.warning("Extraia os dados primeiro na aba de Extração.")

# ─── ETAPA 3: ANÁLISE DE VALOR REAL E CUSTO DE OPORTUNIDADE ──────────────────
elif app_mode == "Análise de Valor Real":
    st.markdown(f"## 💰 Análise de Valor Real: {titulo_dinamico}")
    
    if st.session_state.df_fipe is not None:
        with st.spinner("Consultando dados do Banco Central (IPCA e CDI)..."):
            df_fipe = st.session_state.df_fipe.copy()
            df_ipca = buscar_dados_sgs(433).rename(columns={'valor': 'ipca_mensal'})
            df_cdi = buscar_dados_sgs(4391).rename(columns={'valor': 'cdi_mensal'})
            
            df_ipca['data'] = df_ipca['data'].dt.to_period('M').dt.to_timestamp()
            df_cdi['data'] = df_cdi['data'].dt.to_period('M').dt.to_timestamp()

            df = pd.merge(df_fipe, df_ipca, on='data', how='inner')
            df = pd.merge(df, df_cdi, on='data', how='inner')
            df = df.sort_values('data')

            df['ipca_acumulado'] = (1 + df['ipca_mensal']).cumprod()
            df['ipca_indice_base100'] = df['ipca_acumulado'] * (100 / df['ipca_acumulado'].iloc[0])
            df['indice_real'] = (df['indice'] / df['ipca_indice_base100']) * 100
            
            df['cdi_acumulado'] = (1 + df['cdi_mensal']).cumprod()
            df['indice_cdi_base100'] = df['cdi_acumulado'] * (100 / df['cdi_acumulado'].iloc[0])

            for col in ['indice', 'indice_real']:
                df[f'{col}_n'] = (df[col] / df[col].iloc[0]) * 100

        st.subheader("1. Evolução Imobiliária (Nominal vs. Real)")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df['data'], y=df['indice_real_n'], mode='lines', name='Valor Real (Ajustado IPCA)', line=dict(color='green')))
        fig1.add_trace(go.Scatter(x=df['data'], y=df['indice_n'], mode='lines', name='Valor Nominal', fill='tonexty', fillcolor='rgba(128,128,128,0.2)', line=dict(color='blue')))
        fig1.update_layout(height=400, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("---")
        st.subheader("2. Imóvel (Retorno Total) vs. Benchmarks (CDI)")
        
        yield_anual = st.slider("Taxa de Aluguel Líquido Estimada (% a.a.)", min_value=1.0, max_value=10.0, value=3.5, step=0.1) / 100
        yield_mensal = (1 + yield_anual)**(1/12) - 1

        df['valorizacao_mensal'] = df['indice'].pct_change().fillna(0)
        df['retorno_total_mensal'] = (1 + df['valorizacao_mensal']) * (1 + yield_mensal) - 1
        df['imovel_total_return'] = (1 + df['retorno_total_mensal']).cumprod() * 100
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['data'], y=df['indice_cdi_base100'], mode='lines', name='CDI (Custo de Oportunidade)', line=dict(color='orange', dash='dash', width=2)))
        fig2.add_trace(go.Scatter(x=df['data'], y=df['imovel_total_return'], mode='lines', name=f'Retorno Total (Preço + Aluguel {(yield_anual*100):.1f}% a.a.)', line=dict(color='green', width=3)))
        fig2.add_trace(go.Scatter(x=df['data'], y=df['indice_n'], mode='lines', name='Apenas Valorização Nominal', line=dict(color='blue', width=1.5)))
        
        escala_log = st.checkbox("Usar Escala Logarítmica", value=True)
        if escala_log:
            fig2.update_yaxes(type="log")
            
        fig2.update_layout(height=500, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Evolução do Capital (Base 100)")
        st.plotly_chart(fig2, use_container_width=True)
        
    else: st.warning("Realize a Extração antes de acessar a análise econômica.")

# ─── ETAPA 4: MODELAGEM (SARIMA) ─────────────────────────────────────────────
elif app_mode == "Modelagem":
    st.markdown(f"## 🤖 SARIMA: {titulo_dinamico}")
    
    if st.session_state.df_fipe is not None:
        df_raw = st.session_state.df_fipe.copy()
        df_raw.set_index('data', inplace=True)
        
        with st.sidebar:
            st.markdown("### ⚙️ Configurações da Série")
            min_d, max_d = df_raw.index.min().date(), df_raw.index.max().date()
            start_date, end_date = st.date_input("Período de Análise", [min_d, max_d], min_value=min_d, max_value=max_d)
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
                S = st.number_input("S (Períodos)", 0, 24, 12)
                btn_manual = st.form_submit_button("Aplicar Parâmetros Manuais")
            
            st.markdown("---")
            btn_auto = st.button("🚀 Auto-Fit (pmdarima)", use_container_width=True)

        df = df_raw.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)].copy()
        if suavizar:
            df['indice'] = df['indice'].rolling(window=3, center=True).mean().bfill().ffill()

        train = df['indice'].iloc[:-test_size]
        test = df['indice'].iloc[-test_size:]

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
                if not btn_manual: 
                    order = st.session_state.last_order
                    seasonal_order = st.session_state.last_seasonal
            
            with st.spinner("Treinando modelo..."):
                try:
                    modelo = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False).fit(disp=False)
                    pred_obj = modelo.get_forecast(steps=test_size)
                    pred_df = pred_obj.summary_frame()
                    
                    st.session_state.last_order = order
                    st.session_state.last_seasonal = seasonal_order
                    
                    current_preds = {
                        'mean': pred_df['mean'],
                        'lower': pred_df['mean_ci_lower'],
                        'upper': pred_df['mean_ci_upper'],
                        'label': f"SARIMA {order}x{seasonal_order}"
                    }
                    
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
                    
                    if st.button("📌 Fixar Modelo Atual para Comparação"):
                        st.session_state.pinned_model = current_preds
                        st.success("Modelo fixado! Altere os parâmetros para comparar.")

                    st.markdown("---")
                    st.markdown("### 📈 Real vs Predição")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name='Treino (Completo)', line=dict(color='gray')))
                    fig.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines+markers', name='Real (Teste)', line=dict(color='blue', width=3)))
                    
                    if st.session_state.pinned_model:
                        pinned = st.session_state.pinned_model
                        fig.add_trace(go.Scatter(x=test.index, y=pinned['mean'], mode='lines', name=f"Fixado: {pinned['label']}", line=dict(color='orange', dash='dot')))
                    
                    fig.add_trace(go.Scatter(x=test.index, y=pred_df['mean'], mode='lines', name=f"Atual: {current_preds['label']}", line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=test.index.tolist() + test.index[::-1].tolist(),
                                             y=pred_df['mean_ci_upper'].tolist() + pred_df['mean_ci_lower'][::-1].tolist(),
                                             fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
                                             hoverinfo="skip", showlegend=False, name='Intervalo 95%'))
                    
                    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    st.markdown("### 🔬 Diagnóstico de Resíduos")
                    residuos = modelo.resid[1:]
                    col_hist, col_corr = st.columns([1, 1])
                    with col_hist:
                        fig_hist = go.Figure(data=[go.Histogram(x=residuos, nbinsx=20, marker_color='#3498db')])
                        fig_hist.update_layout(title="Distribuição dos Resíduos (Normalidade)", height=300, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig_hist, use_container_width=True)
                    with col_corr:
                        acf_vals = acf(residuos, nlags=20)
                        pacf_vals = pacf(residuos, nlags=20)
                        fig_corr = make_subplots(rows=2, cols=1, subplot_titles=("Autocorrelação (ACF)", "Autocorrelação Parcial (PACF)"), vertical_spacing=0.15)
                        fig_corr.add_trace(go.Bar(x=np.arange(len(acf_vals)), y=acf_vals, marker_color='red', name='ACF'), row=1, col=1)
                        fig_corr.add_trace(go.Bar(x=np.arange(len(pacf_vals)), y=pacf_vals, marker_color='green', name='PACF'), row=2, col=1)
                        conf_level = 1.96 / np.sqrt(len(residuos))
                        for row in [1, 2]:
                            fig_corr.add_hline(y=conf_level, line_dash="dash", line_color="black", opacity=0.5, row=row, col=1)
                            fig_corr.add_hline(y=-conf_level, line_dash="dash", line_color="black", opacity=0.5, row=row, col=1)
                        fig_corr.update_layout(height=400, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig_corr, use_container_width=True)

                except Exception as e:
                    st.error(f"Erro ao ajustar o modelo com os parâmetros selecionados: {e}")
        else:
            st.info("👆 Selecione os parâmetros manuais ou clique em Auto-Fit para iniciar a modelagem.")
    else: st.error("Realize a Extração antes de acessar a modelagem.")
