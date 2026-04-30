import streamlit as st
import pandas as pd
import io
import asyncio
import subprocess
import sys
from playwright.async_api import async_playwright

# ─── Instala o Chromium do Playwright na primeira execução ──────────────────
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
st.set_page_config(
    page_title="FipeZap · Curitiba",
    page_icon="🏠",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }
    .hero {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        color: white;
    }
    .hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0 0 .4rem; }
    .hero p  { font-size: 1rem; opacity: .75; margin: 0; }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label { font-size: .8rem; color: #64748b; text-transform: uppercase; letter-spacing: .08em; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; font-family: 'Syne', sans-serif; color: #0f2027; }
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #203a43, #2c5364);
        color: white;
        border: none;
        padding: .6rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🏠 FipeZap · Curitiba</h1>
    <p>Extração automatizada do Índice FipeZap — Número Índice / Variação Mensal</p>
</div>
""", unsafe_allow_html=True)

# ─── Scraping ────────────────────────────────────────────────────────────────
async def extrair_fipe_curitiba(info_type: str, log_container):
    async with async_playwright() as p:
        log_container.info("🚀 Iniciando o navegador Chromium...")
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--single-process",
            ],
        )
        
        # Adiciona User-Agent para mascarar o acesso automatizado
        page = await browser.new_page(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        try:
            log_container.info("🌐 Acessando a página da FIPE...")
            await page.goto(
                "https://www.fipe.org.br/pt-br/indices/fipezap/#indice-mensal",
                wait_until="networkidle",
                timeout=60_000,
            )

            log_container.info("🖱️ Preenchendo Tipo: Venda...")
            await page.wait_for_selector("#Tipo_chosen", timeout=20_000)
            await page.click("#Tipo_chosen")
            await page.locator("#Tipo_chosen .chosen-results li").get_by_text("Venda", exact=True).click()
            await page.wait_for_timeout(600)

            log_container.info(f"🖱️ Preenchendo Informação: {info_type}...")
            await page.wait_for_selector("#Info_chosen")
            await page.click("#Info_chosen")
            await page.locator("#Info_chosen .chosen-results li").get_by_text(info_type, exact=True).click()
            await page.wait_for_timeout(600)

            log_container.info("📍 Preenchendo Região: Curitiba...")
            await page.click("#Regiao_chosen")
            await page.locator("#Regiao_chosen input").fill("Curitiba")
            await page.locator("#Regiao_chosen .chosen-results li").get_by_text("Curitiba", exact=True).click()
            await page.wait_for_timeout(600)

            log_container.info("🛏️ Preenchendo Dormitórios: Todos...")
            await page.wait_for_selector("#Dormitorios_chosen")
            await page.click("#Dormitorios_chosen")
            await page.locator("#Dormitorios_chosen .chosen-results li").get_by_text("Todos", exact=True).click()
            await page.wait_for_timeout(400)

            log_container.info("🔍 Clicando em Pesquisar...")
            await page.click("#buttonPesquisar")
            
            log_container.info("⏳ Aguardando a tabela ser gerada pelo site (Procurando botão Imprimir)...")
            await page.wait_for_selector("div.actions a:has-text('Imprimir')", state="visible", timeout=30_000)
            await page.wait_for_timeout(1_200)

            log_container.info("🖨️ Interceptando a janela de impressão...")
            async with page.expect_popup() as popup_info:
                # Usando force=True caso outro elemento esteja levemente sobreposto
                botao = page.locator("div.actions a:has-text('Imprimir')")
                await botao.click(force=True)

            print_page = await popup_info.value
            await print_page.wait_for_load_state("networkidle", timeout=20_000)
            html_content = await print_page.content()

            log_container.info("✅ Lendo o HTML e extraindo tabelas...")
            tabelas = pd.read_html(io.StringIO(html_content))
            await browser.close()
            log_container.empty() # Limpa os logs após o sucesso
            return tabelas

        except Exception as e:
            await browser.close()
            raise e

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    info_type = st.selectbox(
        "Tipo de informação",
        ["Número Índice", "Variação Mensal"],
        index=0,
    )
    st.markdown("---")
    st.markdown("Dados do portal [FIPE](https://www.fipe.org.br).\n\n**Curitiba · Venda · Todos os dormitórios**")

# ─── Main ─────────────────────────────────────────────────────────────────────
col_btn, _ = st.columns([2, 5])
with col_btn:
    run = st.button("🔍 Extrair dados agora", use_container_width=True)

# Contêiner reservado para as mensagens de log da extração
log_placeholder = st.empty()

if run:
    with st.spinner("Executando script de automação..."):
        try:
            tabelas = asyncio.run(extrair_fipe_curitiba(info_type, log_placeholder))
        except Exception as e:
            st.error(f"❌ Erro na extração: {e}")
            tabelas = None

    if tabelas:
        st.success(f"✅ {len(tabelas)} tabela(s) capturada(s)!")

        df = tabelas[0]
        m1, m2, m3 = st.columns(3)
        for col, label, val in zip(
            [m1, m2, m3],
            ["Tabelas", "Linhas", "Colunas"],
            [len(tabelas), len(df), df.shape[1]],
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        for i, t in enumerate(tabelas):
            with st.expander(f"📋 Tabela {i + 1}", expanded=(i == 0)):
                st.dataframe(t, use_container_width=True)
                csv = t.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label=f"⬇️ Baixar Tabela {i + 1} (CSV)",
                    data=csv,
                    file_name=f"fipezap_curitiba_tabela_{i + 1}.csv",
                    mime="text/csv",
                )

        st.markdown("---")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for i, t in enumerate(tabelas):
                t.to_excel(writer, sheet_name=f"Tabela_{i + 1}", index=False)
        st.download_button(
            label="⬇️ Baixar todas as tabelas (Excel)",
            data=buf.getvalue(),
            file_name="fipezap_curitiba_completo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    elif tabelas is not None:
        st.warning("Nenhuma tabela retornada. Tente novamente.")
else:
    st.info("👆 Clique em **Extrair dados agora** para iniciar.")
