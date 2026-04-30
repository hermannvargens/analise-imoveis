#!/bin/bash
# Este script é executado automaticamente pelo Streamlit Cloud após instalar
# os pacotes do requirements.txt.
# Ele baixa o binário do Chromium usado pelo Playwright.
python -m playwright install chromium
