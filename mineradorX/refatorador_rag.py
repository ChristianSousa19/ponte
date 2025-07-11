import os
import re
import nltk
import sys

# --- Nova Constante Configurável ---
# Defina aqui o número mínimo de caracteres que um chunk deve ter.
# Ajuste este valor conforme sua necessidade.
TAMANHO_MINIMO_CHUNK = 300

# --- Configuração Inicial do NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Baixando o pacote 'punkt' do NLTK (necessário para análise de sentenças)...")
    nltk.download('punkt')
    print("Download concluído.")

def aplicar_formatacao_inline(texto):
    """Aplica formatação inline (`) em elementos específicos do texto."""
    texto = re.sub(r'((?<=[\s,(])(/|./)[\w./\-_]+)', r'`\1`', texto)
    texto = re.sub(r'(\$\w+)', r'`\1`', texto)
    texto = re.sub(r'(\b[A-Z_]{3,}=[\w"\./\-_]+)', r'`\1`', texto)
    return texto

def chunkificar_bloco(bloco_texto):
    """
    Aplica as regras de chunking com a nova lógica de acumulação
    para garantir um tamanho mínimo por chunk.
    """
    bloco_texto = aplicar_formatacao_inline(bloco_texto.strip())
    if not bloco_texto:
        return []

    try:
        sentencas = nltk.sent_tokenize(bloco_texto, language='portuguese')
    except LookupError:
        print("Pacote 'punkt' do NLTK parece estar faltando ou incompleto. Baixando/Verificando...")
        nltk.download('punkt')
        print("Download concluído. Tentando tokenizar novamente...")
        sentencas = nltk.sent_tokenize(bloco_texto, language='portuguese')
    except Exception as e:
        print(f"Erro ao tokenizar sentenças, tratando o bloco como um todo: {e}")
        sentencas = [s for s in bloco_texto.split('\n') if s]

    chunks_finais = []
    chunk_temporario = []
    char_count_temporario = 0

    for i, sentenca in enumerate(sentencas):
        sentenca_strip = sentenca.strip()
        
        # Define o que é uma "Quebra Forte": um item de lista principal (1., 2., etc.)
        # mas não um sub-item (1.1., 2.3.1., etc.).
        # Se encontrarmos uma quebra forte e o chunk atual não estiver vazio, forçamos o fechamento.
        is_hard_break = re.match(r'^\s*\d+\.\s', sentenca_strip) and not re.match(r'^\s*\d+\.\d+', sentenca_strip)

        if is_hard_break and chunk_temporario:
            chunks_finais.append(" ".join(chunk_temporario).strip())
            chunk_temporario = []
            char_count_temporario = 0

        # Acumula a sentença atual
        chunk_temporario.append(sentenca)
        char_count_temporario += len(sentenca)

        # Verifica se o chunk atingiu o tamanho e pode ser fechado
        # Condições: (Tamanho > Mínimo E termina com ".") OU (é a última sentença do bloco)
        is_last_sentence = (i == len(sentencas) - 1)
        ends_with_period = sentenca_strip.endswith('.')

        if (char_count_temporario >= TAMANHO_MINIMO_CHUNK and ends_with_period) or is_last_sentence:
            if chunk_temporario:
                chunks_finais.append(" ".join(chunk_temporario).strip())
                chunk_temporario = []
                char_count_temporario = 0

    # Fallback para blocos que não foram chunkificados pelas regras principais
    if not chunks_finais and bloco_texto:
         print("Nenhuma regra de chunking aplicada. Usando fallback da Regra 6 (a cada 10 linhas).")
         linhas = bloco_texto.strip().split('\n')
         for i in range(0, len(linhas), 10):
             pedaco = "\n".join(linhas[i:i+10])
             chunks_finais.append(pedaco)

    return chunks_finais


def processar_arquivo(caminho_arquivo):
    """Função principal que lê, processa e salva o arquivo."""
    print(f"Processando o arquivo: {caminho_arquivo}")
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            conteudo = f.read()
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{caminho_arquivo}' não encontrado.")
        return

    code_blocks = re.findall(r'(```.*?```)', conteudo, re.DOTALL)
    placeholders = []
    for i, block in enumerate(code_blocks):
        placeholder = f"__CODE_BLOCK_PLACEHOLDER_{i}__"
        placeholders.append(placeholder)
        conteudo = conteudo.replace(block, placeholder, 1)

    blocos_iniciais = conteudo.split('###')
    chunks_processados = []
    
    for bloco in blocos_iniciais:
        bloco = bloco.strip()
        if not bloco:
            continue
        novos_chunks = chunkificar_bloco(bloco)
        chunks_processados.extend(novos_chunks)

    conteudo_final = '\n\n###\n\n'.join(chunks_processados)

    for i, placeholder in enumerate(placeholders):
        conteudo_final = conteudo_final.replace(placeholder, code_blocks[i], 1)
        
    base, ext = os.path.splitext(caminho_arquivo)
    caminho_saida = f"{base}_refatorado.txt"
    
    try:
        with open(caminho_saida, 'w', encoding='utf-8') as f:
            f.write(conteudo_final)
        print("\nProcessamento concluído com sucesso!")
        print(f"Arquivo refatorado salvo como: '{caminho_saida}'")
    except Exception as e:
        print(f"ERRO ao salvar o arquivo de saída: {e}")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        nome_arquivo_alvo = sys.argv[1]
    else:
        nome_arquivo_alvo = input("Digite o nome do arquivo de texto a ser processado (ex: meu_arquivo.txt): ")
    
    processar_arquivo(nome_arquivo_alvo)
