import streamlit as st
import pandas as pd
import io
import asyncio
import subprocess
import sys
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
    <p>Extração automatizada e Análise Exploratória de Série Temporal</p>
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
        if df.empty or df.shape[1] < 2:
            continue
        temp_df = df.copy()
        temp_df = temp_df.iloc[:, [0, 1]]
        temp_df.columns = ['mes_str', 'valor']
        temp_df['mes_num'] = temp_df['mes_str'].astype(str).str.lower().str.strip().map(meses_map)
        temp_df = temp_df.dropna(subset=['mes_num'])
        if temp_df.empty:
            continue
        temp_df['mes_num'] = temp_df['mes_num'].astype(int)
        temp_df['Ano'] = ano_corrente
        dfs_processados.append(temp_df)
        ano_corrente -= 1 

    if not dfs_processados:
        raise ValueError("Nenhuma tabela válida de dados foi encontrada após a filtragem.")

    df_final = pd.concat(dfs_processados, ignore_index=True)
    df_final['data'] = pd.to_datetime(df_final['Ano'].astype(str) + '-' + df_final['mes_num'].astype(str) + '-01')
    
    df_series = df_final[['data', 'valor']].copy()
    df_series = df_series.sort_values('data').reset_index(drop=True)
    df_series.columns = ['data', 'indice']
    
    df_series['indice'] = df_series['indice'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df_series['indice'] = df_series['indice'].str.extract(r'(\d+\.?\d*)').astype(float)
    
    return df_series

async def extrair_fipe_curitiba(info_type: str, log_container):
    async with async_playwright() as p:
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

            log_container.info("🔍 Processando pesquisa na FIPE...")
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

run = st.button("🔍 Extrair e Processar Dados", use_container_width=True, type="primary")
log_placeholder = st.empty()

if run:
    with st.spinner("Executando extração. Isso pode levar cerca de 10 segundos..."):
        try:
            tabelas_brutas = asyncio.run(extrair_fipe_curitiba(info_type, log_placeholder))
        except Exception as e:
            st.error(f"❌ Erro: {e}")
            tabelas_brutas = None

    if tabelas_brutas:
        df_temporal = limpar_dados_fipe(tabelas_brutas)
        
        # Prepara a série temporal com index datetime para o Statsmodels
        df_ts = df_temporal.set_index('data')
        
        st.success("✅ Extração e estruturação concluídas!")

        # ─── Separa a visualização em Abas ───
        tab1, tab2 = st.tabs(["📊 Dados Brutos", "📈 Análise Exploratória (EDA)"])

        with tab1:
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="label">Total de Registros (Meses)</div><div class="value">{len(df_temporal)}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="label">Período</div><div class="value">{df_temporal["data"].dt.year.min()} - {df_temporal["data"].dt.year.max()}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df_temporal, use_container_width=True)
            
            csv = df_temporal.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Baixar Série Temporal (CSV)", data=csv, file_name="fipezap_curitiba_serie_temporal.csv", mime="text/csv")

        with tab2:
            st.subheader("1. Evolução Nominal e Retornos")
            
            # Cálculo de variações
            var_total = (df_ts['indice'].iloc[-1] / df_ts['indice'].iloc[0] - 1) * 100
            df_ts['retorno_mensal'] = df_ts['indice'].pct_change() * 100
            
            col1, col2 = st.columns(2)
            col1.metric("Variação Acumulada Total", f"{var_total:.2f}%")
            col2.metric("Último Retorno Mensal", f"{df_ts['retorno_mensal'].iloc[-1]:.2f}%")

            # Gráfico Nativo do Streamlit para Evolução
            st.line_chart(df_ts['indice'])

            st.markdown("---")
            st.subheader("2. Análise de Sazonalidade (Boxplot)")
            
            df_box = df_ts.copy().dropna()
            df_box['mes'] = df_box.index.month
            
            fig_box, ax_box = plt.subplots(figsize=(10, 5))
            df_box.boxplot(column='retorno_mensal', by='mes', ax=ax_box, grid=True)
            plt.title('Distribuição de Retornos Mensais')
            plt.suptitle('') 
            plt.xlabel('Mês')
            plt.ylabel('Variação %')
            st.pyplot(fig_box)

            st.markdown("---")
            st.subheader("3. Decomposição da Série Temporal")
            
            # Decomposição
            decomposicao = seasonal_decompose(df_ts['indice'].dropna(), model='additive', period=12)
            fig_dec = decomposicao.plot()
            fig_dec.set_size_inches(12, 8)
            st.pyplot(fig_dec)

            st.markdown("---")
            st.subheader("4. Análise Estatística para Modelagem (IA/ARIMA)")
            
            col_adf, col_acf = st.columns([1, 1])
            
            with col_adf:
                st.markdown("#### Teste de Estacionariedade (ADF)")
                st.markdown("A série original é estacionária? Modelos como ARIMA e LSTM requerem ou assumem estacionariedade.")
                
                resultado_adf = adfuller(df_ts['indice'].dropna())
                p_value = resultado_adf[1]
                
                st.write(f"**Estatística do Teste:** {resultado_adf[0]:.4f}")
                st.write(f"**P-Value:** {p_value:.4f}")
                
                if p_value < 0.05:
                    st.success("✅ O p-value é menor que 0.05. Rejeitamos a hipótese nula. A série é **Estacionária**.")
                else:
                    st.error("❌ O p-value é maior que 0.05. Falhamos em rejeitar a hipótese nula. A série **NÃO é Estacionária** (Possui raiz unitária). Será necessário aplicar diferenciação (`df.diff()`) antes de modelar.")

            with col_acf:
                st.markdown("#### Autocorrelação (ACF e PACF)")
                fig_acf, (ax_acf, ax_pacf) = plt.subplots(2, 1, figsize=(8, 6))
                
                plot_acf(df_ts['indice'].dropna(), ax=ax_acf, lags=24, title="Função de Autocorrelação (ACF) - Lags p/ MA(q)")
                plot_pacf(df_ts['indice'].dropna(), ax=ax_pacf, lags=24, title="Autocorrelação Parcial (PACF) - Lags p/ AR(p)")
                
                plt.tight_layout()
                st.pyplot(fig_acf)
