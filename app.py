import streamlit as st
import pandas as pd
import io
import asyncio
import subprocess
import sys
from playwright.async_api import async_playwright

# ─── Instala o Chromium na primeira execução ────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Instalando Chromium (apenas na primeira execução)...")
def install_chromium():
    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Falha ao instalar Chromium:\n{result.stderr}")
    return True

install_chromium()

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="FipeZap · Curitiba", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }
    .hero { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); border-radius: 16px; padding: 2.5rem 2rem; margin-bottom: 2rem; color: white; }
    .hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0 0 .4rem; }
    .hero p  { font-size: 1rem; opacity: .75; margin: 0; }
    .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
    .metric-card .label { font-size: .8rem; color: #64748b; text-transform: uppercase; letter-spacing: .08em; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; font-family: 'Syne', sans-serif; color: #0f2027; }
    div[data-testid="stButton"] button { background: linear-gradient(135deg, #203a43, #2c5364); color: white; border: none; padding: .6rem 2rem; font-weight: 600; border-radius: 8px; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🏠 FipeZap · Curitiba</h1>
    <p>Extração automatizada e modelagem de série temporal</p>
</div>
""", unsafe_allow_html=True)

# ─── Funções de Processamento ───────────────────────────────────────────────
def limpar_dados_fipe(lista_tabelas):
    dfs_processados = []
    meses_map = {
        'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
        'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12
    }
    
    ano_corrente = 2026
    
    for df in lista_tabelas:
        temp_df = df.copy()
        temp_df['mes_num'] = temp_df['Mês'].str.lower().map(meses_map)
        temp_df['Ano'] = ano_corrente
        dfs_processados.append(temp_df)
        ano_corrente -= 1 

    df_final = pd.concat(dfs_processados, ignore_index=True)
    df_final['data'] = pd.to_datetime(df_final['Ano'].astype(str) + '-' + df_final['mes_num'].astype(str) + '-01')
    
    df_series = df_final[['data', 'Curitiba']].copy()
    df_series = df_series.sort_values('data').reset_index(drop=True)
    df_series.columns = ['data', 'indice']
    
    # Converte strings com vírgula para floats para permitir operações matemáticas
    df_series['indice'] = df_series['indice'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    
    return df_series

async def extrair_fipe_curitiba(info_type: str, log_container):
    async with async_playwright() as p:
        log_container.info("🚀 Iniciando extração no Chromium...")
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--single-process"]
        )
        
        page = await browser.new_page(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        
        try:
            await page.goto("https://www.fipe.org.br/pt-br/indices/fipezap/#indice-mensal", wait_until="networkidle", timeout=60_000)

            await page.wait_for_selector("#Tipo_chosen", timeout=20_000)
            await page.click("#Tipo_chosen")
            await page.locator("#Tipo_chosen .chosen-results li").get_by_text("Venda", exact=True).click()
            await page.wait_for_timeout(1000)

            await page.click("#Info_chosen")
            await page.locator("#Info_chosen .chosen-results li").get_by_text(info_type, exact=True).click()
            await page.wait_for_timeout(1000)

            await page.click("#Regiao_chosen")
            await page.locator("#Regiao_chosen input").fill("Curitiba")
            await page.wait_for_timeout(500) 
            await page.locator("#Regiao_chosen .chosen-results li").get_by_text("Curitiba", exact=True).click()
            await page.wait_for_timeout(1000)

            await page.click("#Dormitorios_chosen")
            await page.locator("#Dormitorios_chosen .chosen-results li").get_by_text("Todos", exact=True).click()
            await page.wait_for_timeout(1000)

            log_container.info("🔍 Processando pesquisa...")
            await page.click("#buttonPesquisar", force=True)
            
            await page.wait_for_selector("table.results", state="visible", timeout=30_000)
            await page.wait_for_timeout(2000) 

            html_content = await page.content()
            tabelas = pd.read_html(io.StringIO(html_content), attrs={"class": "results"})
            
            await browser.close()
            log_container.empty()
            return tabelas

        except Exception as e:
            await browser.close()
            raise e

# ─── Interface ──────────────────────────────────────────────────────────────
with st.sidebar:
    info_type = st.selectbox("Tipo de informação", ["Número Índice", "Variação Mensal"])

run = st.button("🔍 Extrair e Processar Série Temporal", use_container_width=True, type="primary")
log_placeholder = st.empty()

if run:
    with st.spinner("Executando extração..."):
        try:
            tabelas_brutas = asyncio.run(extrair_fipe_curitiba(info_type, log_placeholder))
        except Exception as e:
            st.error(f"❌ Erro: {e}")
            tabelas_brutas = None

    if tabelas_brutas:
        # Aplica a função de limpeza na lista de tabelas capturadas
        df_temporal = limpar_dados_fipe(tabelas_brutas)
        
        st.success("✅ Série temporal estruturada com sucesso!")

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Total de Registros (Meses)</div>
                <div class="value">{len(df_temporal)}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Período</div>
                <div class="value">{df_temporal['data'].dt.year.min()} - {df_temporal['data'].dt.year.max()}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("### 📈 Visualização")
        # Exibe um gráfico nativo do Streamlit utilizando a data como índice
        st.line_chart(df_temporal.set_index('data'), y='indice')

        st.markdown("### 📋 DataFrame Final")
        st.dataframe(df_temporal, use_container_width=True)
        
        csv = df_temporal.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Baixar Série Temporal (CSV)",
            data=csv,
            file_name="fipezap_curitiba_serie_temporal.csv",
            mime="text/csv",
        )
