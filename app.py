import streamlit as st
import pandas as pd
import numpy as np
import io
import asyncio
import subprocess
import sys
import requests
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

# ─── Mapeamento de Cidades e Dormitórios ─────────────────────────────────────
CIDADES = {
    "Aracaju": "Aracaju", "Balneário Camboriú": "Balneário Camboriú", "Barueri": "Barueri",
    "Belém": "Belém", "Belo Horizonte": "Belo Horizonte", "Betim": "Betim",
    "Blumenau": "Blumenau", "Campinas": "Campinas", "Campo Grande": "Campo Grande",
    "Canoas": "Canoas", "Caxias do Sul": "Caxias do Sul", "Contagem": "Contagem",
    "Cuiabá": "Cuiabá", "Curitiba": "Curitiba", "Diadema": "Diadema",
    "Distrito Federal": "Distrito Federal", "Florianópolis": "Florianópolis", "Fortaleza": "Fortaleza",
    "Goiânia": "Goiânia", "Guarujá": "Guarujá", "Guarulhos": "Guarulhos",
    "Itajaí": "Itajaí", "Itapema": "Itapema", "Jaboatão dos Guararapes": "Jaboatão dos Guararapes",
    "João Pessoa": "João Pessoa", "Joinville": "Joinville", "Londrina": "Londrina",
    "Maceió": "Maceió", "Manaus": "Manaus", "Natal": "Natal", "Niterói": "Niterói",
    "Novo Hamburgo": "Novo Hamburgo", "Osasco": "Osasco", "Pelotas": "Pelotas",
    "Porto Alegre": "Porto Alegre", "Praia Grande": "Praia Grande", "Recife": "Recife",
    "Ribeirão Preto": "Ribeirão Preto", "Rio de Janeiro": "Rio de Janeiro", "Salvador": "Salvador",
    "Santa Maria": "Santa Maria", "Santo André": "Santo André", "Santos": "Santos",
    "São Bernardo do Campo": "São Bernardo do Campo", "São Caetano do Sul": "São Caetano do Sul",
    "São José": "São José", "São José do Rio Preto": "São José do Rio Preto",
    "São José dos Campos": "São José dos Campos", "São José dos Pinhais": "São José dos Pinhais",
    "São Leopoldo": "São Leopoldo", "São Luís": "São Luís", "São Paulo": "São Paulo",
    "São Vicente": "São Vicente", "Teresina": "Teresina", "Vila Velha": "Vila Velha", "Vitória": "Vitória"
}

DORMITORIOS = {"Todos": "Todos", "1": "1", "2": "2", "3": "3", "4": "4"}

# ─── Funções de Suporte ──────────────────────────────────────────────────────
def limpar_dados_fipe(lista_tabelas, cidade_nome):
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

async def extrair_fipe_dinamico(cidade, quartos, log_container):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
        page = await browser.new_page(user_agent="Mozilla/5.0")
        try:
            await page.goto("https://www.fipe.org.br/pt-br/indices/fipezap/#indice-mensal", wait_until="networkidle")
            
            # Tipo: Venda
            await page.click("#Tipo_chosen")
            await page.locator("#Tipo_chosen .chosen-results li").get_by_text("Venda", exact=True).click()
            await page.wait_for_timeout(600)

            # Info: Número Índice
            await page.click("#Info_chosen")
            await page.locator("#Info_chosen .chosen-results li").get_by_text("Número Índice", exact=True).click()
            await page.wait_for_timeout(600)

            # Região: Dinâmico
            await page.click("#Regiao_chosen")
            await page.locator("#Regiao_chosen input").fill(cidade)
            await page.wait_for_timeout(500)
            await page.locator("#Regiao_chosen .chosen-results li").get_by_text(cidade, exact=True).click()
            await page.wait_for_timeout(600)

            # Dormitórios: Dinâmico
            await page.click("#Dormitorios_chosen")
            await page.locator("#Dormitorios_chosen .chosen-results li").get_by_text(quartos, exact=True).click()
            
            log_container.info(f"🔍 Consultando base FIPE: {cidade} ({quartos} quartos)...")
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

# ─── Inicialização e Sidebar ──────────────────────────────────────────────────
st.set_page_config(page_title="FipeZap Analytics", page_icon="🏠", layout="wide")

with st.sidebar:
    st.title("🏠 Menu")
    app_mode = st.radio("Selecione uma etapa:", ["Extração", "Análise Exploratória (EDA)", "Análise de Valor Real", "Modelagem"])
    st.markdown("---")
    
    st.markdown("### 📍 Filtros de Extração")
    cidade_selecionada = st.selectbox("Cidade", options=list(CIDADES.keys()), index=13) # Default Curitiba
    quartos_selecionados = st.selectbox("Quantidade de Dormitórios", options=list(DORMITORIOS.keys()), index=0) # Default Todos

if 'df_fipe' not in st.session_state:
    st.session_state.df_fipe = None
if 'pinned_model' not in st.session_state:
    st.session_state.pinned_model = None

# ─── ETAPA 1: EXTRAÇÃO ───────────────────────────────────────────────────────
if app_mode == "Extração":
    st.markdown(f"## 📥 Extração de Dados: {cidade_selecionada}")
    col_btn, _ = st.columns([2, 5])
    with col_btn:
        btn_extract = st.button("🚀 Iniciar Scraping", use_container_width=True, type="primary")
    
    log_place = st.empty()
    if btn_extract:
        with st.spinner(f"Acessando portal FIPE para {cidade_selecionada}..."):
            try:
                tabelas = asyncio.run(extrair_fipe_dinamico(cidade_selecionada, quartos_selecionados, log_place))
                st.session_state.df_fipe = limpar_dados_fipe(tabelas, cidade_selecionada)
                st.success(f"Dados de {cidade_selecionada} prontos!")
            except Exception as e: st.error(f"Erro: {e}")
            
    if st.session_state.df_fipe is not None:
        st.dataframe(st.session_state.df_fipe, use_container_width=True)

# ─── (As demais etapas EDA, Valor Real e Modelagem permanecem as mesmas) ─────
# Elas usarão automaticamente o st.session_state.df_fipe que foi extraído.
# Para economizar espaço, o restante do código segue a lógica anterior.
