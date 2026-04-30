# 🏠 FipeZap Curitiba — Streamlit App

Extração automatizada do **Índice FipeZap** para **Curitiba** usando Playwright (Chromium headless) e Streamlit.

## 📁 Estrutura do repositório

```
├── app.py                  # Aplicativo principal
├── requirements.txt        # Dependências Python
├── packages.txt            # Dependências de sistema (Ubuntu)
├── postinstall.sh          # Baixa o Chromium do Playwright após deploy
└── .streamlit/
    └── config.toml         # Configurações do Streamlit
```

## 🚀 Deploy no Streamlit Community Cloud (gratuito)

### Passo a passo

1. **Fork / clone** este repositório para sua conta do GitHub.

2. Acesse [share.streamlit.io](https://share.streamlit.io) e clique em **New app**.

3. Selecione:
   - **Repository**: seu repositório
   - **Branch**: `main`
   - **Main file path**: `app.py`

4. Clique em **Deploy!** — o Streamlit Cloud irá:
   - Instalar os pacotes de sistema do `packages.txt`
   - Instalar as libs Python do `requirements.txt`
   - Rodar o `postinstall.sh` para baixar o Chromium

5. Aguarde ~3 minutos. Acesse o link público gerado.

### ⚠️ Por que Chromium e não Firefox?

O Streamlit Community Cloud usa Ubuntu 22.04. O **Firefox** requer dependências de sistema (`libgtk-3`) que **não estão disponíveis** no ambiente gratuito. O **Chromium** funciona perfeitamente com as libs listadas em `packages.txt`.

## 🔧 Rodando localmente

```bash
# 1. Clone o repo
git clone https://github.com/SEU_USUARIO/SEU_REPO.git
cd SEU_REPO

# 2. Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Baixe o Chromium
python -m playwright install chromium

# 5. Rode o app
streamlit run app.py
```

## 📊 Funcionalidades

- Extrai automaticamente as tabelas do [FipeZap](https://www.fipe.org.br/pt-br/indices/fipezap/)
- Filtra por: **Venda · Curitiba · Todos os dormitórios**
- Permite escolher entre **Número Índice** ou **Variação Mensal**
- Exibe as tabelas interativamente no browser
- Download individual por tabela (CSV) ou download completo (Excel)
