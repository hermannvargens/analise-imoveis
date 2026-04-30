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
    .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem; text-align: center; }
    .metric-card .label { font-size: .8rem; color: #64748b; text-transform: uppercase; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #0f2027; }
</style>
""", unsafe_allow_html=True)

st.title("🏠 Extração FipeZap - Curitiba")

# ─── Scraping com Captura de Tela ───────────────────────────────────────────
async def extrair_fipe_curitiba(info_type: str, log_container, image_container):
    async with async_playwright() as p:
        log_container.info("🚀 Iniciando o navegador Chromium...")
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--single-process"]
        )
        
        page = await browser.new_page(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800} # Define um tamanho fixo para a foto ficar boa
        )
        
        try:
            log_container.info("🌐 Acessando a página da FIPE...")
            await page.goto("https://www.fipe.org.br/pt-br/indices/fipezap/#indice-mensal", wait_until="networkidle", timeout=60_000)

            # Preenchimento
            log_container.info("🖱️ Preenchendo Tipo: Venda...")
            await page.wait_for_selector("#Tipo_chosen", timeout=20_000)
            await page.click("#Tipo_chosen")
            await page.locator("#Tipo_chosen .chosen-results li").get_by_text("Venda", exact=True).click()
            await page.wait_for_timeout(800) # Aumentei levemente as pausas

            log_container.info(f"🖱️ Preenchendo Informação: {info_type}...")
            await page.click("#Info_chosen")
            await page.locator("#Info_chosen .chosen-results li").get_by_text(info_type, exact=True).click()
            await page.wait_for_timeout(800)

            log_container.info("📍 Preenchendo Região: Curitiba...")
            await page.click("#Regiao_chosen")
            await page.locator("#Regiao_chosen input").fill("Curitiba")
            await page.locator("#Regiao_chosen .chosen-results li").get_by_text("Curitiba", exact=True).click()
            await page.wait_for_timeout(800)

            log_container.info("🛏️ Preenchendo Dormitórios: Todos...")
            await page.click("#Dormitorios_chosen")
            await page.locator("#Dormitorios_chosen .chosen-results li").get_by_text("Todos", exact=True).click()
            await page.wait_for_timeout(800)

            # --- O PULO DO GATO: CAPTURA DE TELA AQUI ---
            log_container.info("📸 Tirando print de verificação...")
            screenshot_bytes = await page.screenshot(full_page=True)
            image_container.image(screenshot_bytes, caption="VISÃO DO ROBÔ: Formulário instantes antes do clique na pesquisa")

            log_container.info("🔍 Clicando em Pesquisar (forçando o clique caso algo esteja na frente)...")
            await page.click("#buttonPesquisar", force=True) # force=True ignora se algo estiver cobrindo o botão
            
            log_container.info("⏳ Aguardando tabela...")
            await page.wait_for_selector("div.actions a:has-text('Imprimir')", state="visible", timeout=30_000)
            await page.wait_for_timeout(1_200)

            async with page.expect_popup() as popup_info:
                botao = page.locator("div.actions a:has-text('Imprimir')")
                await botao.click(force=True)

            print_page = await popup_info.value
            await print_page.wait_for_load_state("networkidle", timeout=20_000)
            html_content = await print_page.content()

            tabelas = pd.read_html(io.StringIO(html_content))
            await browser.close()
            log_container.empty()
            return tabelas

        except Exception as e:
            # Se der erro, tira outro print para vermos a tela de erro da FIPE
            log_container.error("Erro detectado. Capturando a tela final...")
            erro_bytes = await page.screenshot(full_page=True)
            image_container.image(erro_bytes, caption="VISÃO DO ROBÔ: Momento em que o erro ocorreu")
            await browser.close()
            raise e

# ─── Interface ──────────────────────────────────────────────────────────────
with st.sidebar:
    info_type = st.selectbox("Tipo de informação", ["Número Índice", "Variação Mensal"])

run = st.button("🔍 Extrair dados agora", use_container_width=True, type="primary")

# Espaços reservados para atualizar a interface durante o processo assíncrono
log_placeholder = st.empty()
image_placeholder = st.empty()

if run:
    with st.spinner("Executando automação..."):
        try:
            tabelas = asyncio.run(extrair_fipe_curitiba(info_type, log_placeholder, image_placeholder))
        except Exception as e:
            st.error(f"❌ Erro: {e}")
            tabelas = None

    if tabelas:
        st.success(f"✅ {len(tabelas)} tabela(s) capturada(s)!")
        st.dataframe(tabelas[0], use_container_width=True)
