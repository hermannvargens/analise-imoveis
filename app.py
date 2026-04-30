import streamlit as st
import pandas as pd
import io
import asyncio
from playwright.async_api import async_playwright

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FipeZap · Curitiba",
    page_icon="🏠",
    layout="wide",
)

# ─── Minimal custom CSS ─────────────────────────────────────────────────────
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
        transition: opacity .2s;
    }
    div[data-testid="stButton"] button:hover { opacity: .85; }
</style>
""", unsafe_allow_html=True)

# ─── Hero header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏠 FipeZap · Curitiba</h1>
    <p>Extração automatizada do Índice FipeZap — Número Índice / Variação Mensal</p>
</div>
""", unsafe_allow_html=True)

# ─── Scraping function ───────────────────────────────────────────────────────
async def extrair_fipe_curitiba(info_type: str = "Número Índice") -> list[pd.DataFrame] | None:
    """
    Extrai tabelas do FipeZap para Curitiba usando Chromium headless.
    info_type: 'Número Índice' ou 'Variação Mensal'
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        page = await browser.new_page()

        try:
            await page.goto(
                "https://www.fipe.org.br/pt-br/indices/fipezap/#indice-mensal",
                wait_until="networkidle",
                timeout=60_000,
            )

            # 1. Tipo → Venda
            await page.wait_for_selector("#Tipo_chosen", timeout=20_000)
            await page.click("#Tipo_chosen")
            await page.locator("#Tipo_chosen .chosen-results li").get_by_text("Venda", exact=True).click()
            await page.wait_for_timeout(600)

            # 2. Informação
            await page.wait_for_selector("#Info_chosen")
            await page.click("#Info_chosen")
            await page.locator("#Info_chosen .chosen-results li").get_by_text(info_type, exact=True).click()
            await page.wait_for_timeout(600)

            # 3. Região → Curitiba
            await page.click("#Regiao_chosen")
            await page.locator("#Regiao_chosen input").fill("Curitiba")
            await page.locator("#Regiao_chosen .chosen-results li").get_by_text("Curitiba", exact=True).click()
            await page.wait_for_timeout(600)

            # 4. Dormitórios → Todos
            await page.wait_for_selector("#Dormitorios_chosen")
            await page.click("#Dormitorios_chosen")
            await page.locator("#Dormitorios_chosen .chosen-results li").get_by_text("Todos", exact=True).click()
            await page.wait_for_timeout(400)

            # 5. Pesquisar
            await page.click("#buttonPesquisar")
            await page.wait_for_selector("div.actions a:has-text('Imprimir')", timeout=30_000)
            await page.wait_for_timeout(1_200)

            # 6. Popup de impressão → capturar HTML
            async with page.expect_popup() as popup_info:
                botao = page.locator("div.actions a:has-text('Imprimir')")
                await botao.dispatch_event("click")

            print_page = await popup_info.value
            await print_page.wait_for_load_state("networkidle", timeout=20_000)
            html_content = await print_page.content()

            tabelas = pd.read_html(io.StringIO(html_content))
            await browser.close()
            return tabelas

        except Exception as e:
            await browser.close()
            raise e


# ─── Sidebar controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    info_type = st.selectbox(
        "Tipo de informação",
        ["Número Índice", "Variação Mensal"],
        index=0,
    )
    st.markdown("---")
    st.markdown(
        "Dados extraídos diretamente do portal [FIPE](https://www.fipe.org.br).\n\n"
        "Região: **Curitiba** · Segmento: **Venda** · Dormitórios: **Todos**"
    )

# ─── Main panel ─────────────────────────────────────────────────────────────
col_btn, col_info = st.columns([2, 5])
with col_btn:
    run = st.button("🔍 Extrair dados agora", use_container_width=True)

if run:
    with st.spinner("Acessando o portal FipeZap via Chromium… aguarde (~30 s)"):
        try:
            tabelas = asyncio.run(extrair_fipe_curitiba(info_type))
        except Exception as e:
            st.error(f"❌ Erro na extração: {e}")
            tabelas = None

    if tabelas:
        st.success(f"✅ {len(tabelas)} tabela(s) capturada(s) com sucesso!")

        # ── Métricas rápidas da primeira tabela ──────────────────────────────
        df = tabelas[0]

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Tabelas capturadas</div>
                <div class="value">{len(tabelas)}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Linhas (tabela 1)</div>
                <div class="value">{len(df)}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Colunas (tabela 1)</div>
                <div class="value">{df.shape[1]}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Exibir tabelas ────────────────────────────────────────────────────
        for i, t in enumerate(tabelas):
            with st.expander(f"📋 Tabela {i + 1}", expanded=(i == 0)):
                st.dataframe(t, use_container_width=True)

                # Download CSV individual
                csv = t.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label=f"⬇️ Baixar Tabela {i + 1} (CSV)",
                    data=csv,
                    file_name=f"fipezap_curitiba_tabela_{i + 1}.csv",
                    mime="text/csv",
                )

        # ── Download de todas as tabelas em um Excel ──────────────────────────
        st.markdown("---")
        st.markdown("#### 📦 Download completo")
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
    else:
        st.warning("Nenhuma tabela foi retornada. Tente novamente.")

else:
    st.info("👆 Clique em **Extrair dados agora** para iniciar o scraping do portal FipeZap.")
