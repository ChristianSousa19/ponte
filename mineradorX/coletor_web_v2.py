# Importamos as bibliotecas necessárias
import os
import time
from datetime import datetime
from typing import List, Optional
import undetected_chromedriver as uc

from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# --- CONFIGURAÇÃO E GERENCIAMENTO DO DRIVER (sem alterações) ---

def setup_driver() -> uc.Chrome:
    print("-> Configurando o navegador em modo 'Stealth'...")
    chrome_options = uc.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
    
    try:
        driver = uc.Chrome(options=chrome_options, use_subprocess=True)
        print("✅ Navegador 'Stealth' pronto.")
        return driver
    except WebDriverException as e:
        print(f"❌ ERRO CRÍTICO: Não foi possível iniciar o driver do Chrome. Verifique a instalação.")
        print(f"   Detalhe do erro: {e}")
        exit()

# --- LÓGICA DE NAVEGAÇÃO E EXTRAÇÃO ---

def get_page_content_selenium(driver: uc.Chrome, url: str) -> Optional[str]:
    # Esta função não precisa de alterações
    try:
        print(f"-> Navegando para: {url}")
        driver.get(url)
        print("-> Aguardando a renderização da página...")
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        print("   Página carregada.")
        time.sleep(3) 
        return driver.page_source
    except TimeoutException:
        print(f"❌ ERRO: Timeout ao esperar a página carregar: {url}")
        return None
    except WebDriverException as e:
        print(f"❌ ERRO: Falha ao acessar a URL com Selenium: {url}. Erro: {e}")
        return None

def extract_text_from_html(html_content: Optional[str]) -> str:
    if not html_content:
        return ""
        
    soup = BeautifulSoup(html_content, 'html.parser')

    for element in soup(["script", "style", "header", "footer", "nav", "aside", "form", "button", "iframe"]):
        element.decompose()

    main_content = soup.find('article') or soup.find('main') or soup.find('div', class_=lambda c: c and 'content' in c)
    
    if main_content:
        text_blocks = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4'])
    else:
        text_blocks = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4'])
    
    ### CORREÇÃO ###
    # Trocamos "\n\n" por "\n" para juntar as linhas de texto sem uma linha em branco entre elas.
    full_text = "\n".join(block.get_text(strip=True) for block in text_blocks if block.get_text(strip=True))
    
    return full_text

def attempt_paywall_removal(driver: uc.Chrome) -> Optional[str]:
    # Esta função não precisa de alterações
    print("⚠️ Tentando remover overlays de paywall com JavaScript...")
    try:
        driver.execute_script("""
            var elements = document.querySelectorAll('[id*="paywall"], [class*="paywall"], [id*="overlay"], [class*="overlay"]');
            elements.forEach(function(el) { el.style.display = "none"; });
        """)
        driver.execute_script("document.body.style.overflow = 'auto';")
        time.sleep(2)
        print("✅ Scripts de remoção de overlay executados.")
        return driver.page_source
    except WebDriverException as js_error:
        print(f"❌ Falha ao tentar executar remoção de overlay: {js_error}")
        return None

# --- FUNÇÕES DE ARQUIVO E ORQUESTRAÇÃO ---

def save_content_to_file(content: str) -> None:
    # Esta função não precisa de alterações
    if not content:
        print("\nNenhum conteúdo foi coletado para salvar.")
        return

    try:
        filename = f"base_conhecimento_web_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        print(f"\n💾 Salvando todo o conteúdo limpo no arquivo: {filename}")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content.strip())
            
        print("\n✅ Missão concluída! Sua nova base de conhecimento web está pronta.")
    except IOError as e:
        print(f"❌ ERRO: Não foi possível salvar o arquivo '{filename}'. Erro: {e}")

def get_urls_from_user() -> List[str]:
    # Esta função não precisa de alterações
    user_input = input("\nPor favor, insira URLs (separadas por vírgula) ou o caminho de um arquivo .txt: ").strip()
    urls_string = ""
    if user_input.lower().endswith(".txt") and os.path.isfile(user_input):
        print(f"-> Lendo URLs do arquivo: {user_input}")
        try:
            with open(user_input, 'r', encoding='utf-8') as f:
                urls_string = f.read()
            print("✅ Arquivo lido com sucesso.")
        except IOError as e:
            print(f"❌ ERRO: Não foi possível ler o arquivo '{user_input}'. Erro: {e}")
            return []
    else:
        urls_string = user_input

    if not urls_string:
        return []

    urls = [url.strip() for url in urls_string.replace('\n', ',').split(',') if url.strip()]
    return urls

def coletar_noticias_web() -> None:
    print("--- Assistente de Coleta de Notícias v9 (Saída Compacta) ---")
    
    urls = get_urls_from_user()
    
    if not urls:
        print("Nenhuma URL válida foi fornecida. Encerrando.")
        return

    all_articles_content = ""

    with setup_driver() as driver:
        for i, url in enumerate(urls):
            if not url.startswith(('http://', 'https://')):
                print(f"\n⚠️ Ignorando entrada inválida: '{url}'")
                continue

            print(f"\n--- Processando URL {i+1}/{len(urls)} ---")
            
            html_content = get_page_content_selenium(driver, url)
            cleaned_text = extract_text_from_html(html_content)
            
            if not cleaned_text or len(cleaned_text) < 200:
                html_content_after_js = attempt_paywall_removal(driver)
                cleaned_text = extract_text_from_html(html_content_after_js)

            if cleaned_text:
                print("✅ Texto extraído com sucesso.")
                ### CORREÇÃO ###
                # Reduzimos os espaçamentos para tornar a saída mais compacta.
                # Usamos \n em vez de \n\n e ajustamos o separador.
                all_articles_content += f"### Fonte da Notícia: {url}\n"
                all_articles_content += f"# Data da Coleta: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                all_articles_content += cleaned_text + "\n\n---\n\n" # Deixamos um espaçamento maior apenas no separador de artigos.
            else:
                print("❌ Falha final: Não foi possível extrair texto significativo desta URL.")

    save_content_to_file(all_articles_content)

if __name__ == '__main__':
    coletar_noticias_web()
