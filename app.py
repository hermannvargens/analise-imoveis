import streamlit as st
import pandas as pd
import io
import asyncio
import subprocess
import sys
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from playwright.async_api import async_playwright

# ─── Instalação do Chromium ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Preparando ambiente (Chromium)...")
def install_chromium():
    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        capture_output=True, text=True
    )
    return True

install_chromium()

# ─── Funções de Scraping e Limpeza ──────────────────────────────────────────
def limpar_dados_fipe(lista_tabelas):
    dfs_processados = []
    meses_map = {
        'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
        'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12
    }
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
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu", "--single-process"])
        page = await browser.new_page(user_agent="Mozilla/5.0")
        try:
            await page.goto("https://www.fipe.org.br/pt-br/indices/fipezap/#indice-mensal", wait_until="networkidle")
            
            # Preenchimento padrão (Venda / Número Índice / Curitiba / Todos)
            await page.click("#Tipo_chosen")
            await page.locator("#Tipo_chosen .chosen-results li").get_by_text("Venda", exact=True).click()
            await page.wait_for_timeout(800)

            await page.click("#Info_chosen")
            await page.locator("#Info_chosen .chosen-results li").get_by_text("Número Índice", exact=True).click()
            await page.wait_for_timeout(800)

            await page.click("#Regiao_chosen")
            await page.locator("#Regiao_chosen input").fill("Curitiba")
            await page.wait_for_timeout(500)
            await page.locator("#Regiao_chosen .chosen-results li").get_by_text("Curitiba", exact=True).click()
            await page.wait_for_timeout(800)

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

# ─── Configuração de Página e Sidebar ────────────────────────────────────────
st.set_page_config(page_title="FipeZap Analytics", page_icon="🏠", layout="wide")

with st.sidebar:
    st.title("🏠 Menu")
    app_mode = st.radio("Selecione uma etapa:", ["Extração", "Análise Exploratória (EDA)", "Modelagem"])
    st.markdown("---")
    st.info("Foco: Curitiba | Venda | Número Índice")

# Inicialização de estado para os dados
if 'df_fipe' not in st.session_state:
    st.session_state.df_fipe = None

# ─── ETAPA 1: EXTRAÇÃO ───────────────────────────────────────────────────────
if app_mode == "Extração":
    st.markdown('<div style="background: linear-gradient(135deg, #0f2027, #203a43); padding: 2rem; border-radius: 15px; color: white;"><h1>Extração de Dados</h1><p>Obtenção automática dos dados brutos do portal FipeZap.</p></div>', unsafe_allow_html=True)
    
    col_btn, _ = st.columns([2, 5])
    with col_btn:
        btn_extract = st.button("🚀 Iniciar Scraping", use_container_width=True, type="primary")
    
    log_place = st.empty()
    
    if btn_extract:
        with st.spinner("Conectando ao portal FIPE..."):
            try:
                tabelas = asyncio.run(extrair_fipe_curitiba(log_place))
                st.session_state.df_fipe = limpar_dados_fipe(tabelas)
                st.success("Dados extraídos e estruturados com sucesso!")
            except Exception as e:
                st.error(f"Erro na extração: {e}")

    if st.session_state.df_fipe is not None:
        st.subheader("Visualização dos Dados Estruturados")
        st.dataframe(st.session_state.df_fipe, use_container_width=True)
        csv = st.session_state.df_fipe.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Baixar CSV", csv, "fipe_curitiba.csv", "text/csv")
    else:
        st.warning("Aguardando extração para exibir dados.")

# ─── ETAPA 2: EDA ────────────────────────────────────────────────────────────
elif app_mode == "Análise Exploratória (EDA)":
    st.markdown('<div style="background: linear-gradient(135deg, #2c3e50, #000000); padding: 2rem; border-radius: 15px; color: white;"><h1>Análise Exploratória</h1><p>Decomposição, Sazonalidade e Tendência.</p></div>', unsafe_allow_html=True)
    
    if st.session_state.df_fipe is not None:
        df = st.session_state.df_fipe.set_index('data')
        
        # Métricas Rápidas
        var_total = (df['indice'].iloc[-1] / df['indice'].iloc[0] - 1) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("Início da Série", df.index.min().strftime('%m/%Y'))
        m2.metric("Fim da Série", df.index.max().strftime('%m/%Y'))
        m3.metric("Variação Acumulada", f"{var_total:.2f}%")

        st.markdown("### Evolução Histórica (Nominal)")
        st.line_chart(df['indice'])

        st.markdown("---")
        col_box, col_dec = st.columns([1, 1])

        with col_box:
            st.subheader("Sazonalidade Mensal")
            df_ret = df.copy()
            df_ret['retorno'] = df_ret['indice'].pct_change() * 100
            df_ret['mes'] = df_ret.index.month
            fig_box, ax_box = plt.subplots()
            df_ret.dropna().boxplot(column='retorno', by='mes', ax=ax_box)
            ax_box.set_title("Variação % por Mês")
            plt.suptitle("")
            st.pyplot(fig_box)

        with col_dec:
            st.subheader("Decomposição (Componentes)")
            dec = seasonal_decompose(df['indice'], model='additive', period=12)
            fig_dec, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(dec.trend, color='orange', label='Tendência')
            axes[0].legend()
            axes[1].plot(dec.seasonal, color='green', label='Sazonalidade')
            axes[1].legend()
            axes[2].scatter(df.index, dec.resid, alpha=0.5, label='Resíduo')
            axes[2].legend()
            st.pyplot(fig_dec)
    else:
        st.error("Por favor, realize a 'Extração' de dados antes de acessar a análise.")

# ─── ETAPA 3: MODELAGEM ──────────────────────────────────────────────────────
elif app_mode == "Modelagem":
    st.title("🤖 Modelagem Preditiva")
    if st.session_state.df_fipe is not None:
        st.info("Espaço reservado para implementação dos modelos SARIMAX, LSTM e XGBoost.")
        st.write("Dados prontos para processamento:", st.session_state.df_fipe.shape)
        # Próximos passos: Teste de Estacionariedade, Train/Test Split, etc.
    else:
        st.error("Dados não encontrados para modelagem.")
