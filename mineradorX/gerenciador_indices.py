import os
import json
import shutil
import time
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Importa nosso chunker
from chunker_customizado import chunkificar_texto_completo

# --- CONFIGURAÇÃO ---
load_dotenv()
print("-> Carregando modelo de embedding (isso pode levar um momento)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Modelo de embedding carregado.")

# --- FUNÇÕES DE CARREGAMENTO DE DADOS ---

def carregar_fontes(lista_fontes: list[str]) -> list[Document]:
    """
    Carrega o conteúdo de diferentes tipos de fontes (URL, PDF, TXT).
    """
    documentos_totais = []
    for fonte in lista_fontes:
        print(f"   -> Carregando fonte: {fonte}")
        try:
            if fonte.startswith(('http://', 'https://')):
                loader = WebBaseLoader(fonte)
                documentos_totais.extend(loader.load())
            elif fonte.lower().endswith('.pdf'):
                loader = PyPDFLoader(fonte)
                documentos_totais.extend(loader.load())
            elif fonte.lower().endswith('.txt'):
                loader = TextLoader(fonte, encoding='utf-8')
                documentos_totais.extend(loader.load())
            else:
                print(f"      ⚠️ Tipo de arquivo não suportado: {fonte}")
        except Exception as e:
            print(f"      ❌ Erro ao carregar a fonte {fonte}: {e}")
    return documentos_totais

# --- FUNÇÕES DE GERENCIAMENTO DE ÍNDICE ---

def criar_ou_atualizar_contexto(contexto_id: str, definicoes_contexto: dict):
    """
    Cria ou atualiza o índice para um contexto específico.
    """
    pasta_base_indices = "indices_rag"
    pasta_contexto = os.path.join(pasta_base_indices, contexto_id)

    if os.path.exists(pasta_contexto):
        print(f"-> Contexto '{contexto_id}' já existe. Removendo índice antigo para atualização.")
        shutil.rmtree(pasta_contexto)

    print(f"-> Criando novo índice para o contexto: '{contexto_id}'")
    
    # 1. Carregar todas as fontes
    documentos_carregados = carregar_fontes(definicoes_contexto["fontes"])
    if not documentos_carregados:
        print(f"❌ Nenhuma fonte válida encontrada ou carregada para '{contexto_id}'. Abortando.")
        return

    texto_completo = "\n\n".join([doc.page_content for doc in documentos_carregados])

    # 2. Usar nosso chunker customizado
    print("   -> Dividindo o texto com o chunker customizado...")
    chunks_de_texto = chunkificar_texto_completo(texto_completo)
    docs_para_indexar = [Document(page_content=chunk) for chunk in chunks_de_texto]
    print(f"   -> Texto dividido em {len(docs_para_indexar)} chunks.")

    # 3. Criar e salvar o índice FAISS
    print("   -> Criando embeddings e salvando o índice...")
    start_time = time.time()
    db = FAISS.from_documents(docs_para_indexar, embeddings)
    os.makedirs(pasta_contexto, exist_ok=True)
    db.save_local(pasta_contexto)
    end_time = time.time()
    
    print(f"✅ Índice para '{contexto_id}' criado com sucesso em '{pasta_contexto}'. (Levou {end_time - start_time:.2f} segundos)")

def deletar_contexto(contexto_id: str):
    """
    Deleta a pasta de índice de um contexto.
    """
    pasta_base_indices = "indices_rag"
    pasta_contexto = os.path.join(pasta_base_indices, contexto_id)

    if os.path.exists(pasta_contexto):
        print(f"-> Deletando índice do contexto '{contexto_id}'...")
        shutil.rmtree(pasta_contexto)
        print("✅ Índice deletado com sucesso.")
    else:
        print(f"⚠️ Índice para o contexto '{contexto_id}' não encontrado. Nada a fazer.")


# --- EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerenciador de Índices RAG por Contexto.")
    parser.add_argument("--acao", type=str, required=True, choices=['criar', 'deletar'], help="A ação a ser executada. criar ou deletar.")
    parser.add_argument("--contexto", type=str, required=True, help="O ID do contexto (ex: 'futebol') definido em contexts.json.")

    args = parser.parse_args()

    try:
        with open("contexts.json", 'r', encoding='utf-8') as f:
            todos_contextos = json.load(f)
    except FileNotFoundError:
        print("ERRO: Arquivo 'contexts.json' não encontrado. Crie-o antes de continuar.")
        exit()

    if args.contexto not in todos_contextos:
        print(f"ERRO: Contexto '{args.contexto}' não definido em contexts.json.")
        exit()

    if args.acao == 'criar':
        criar_ou_atualizar_contexto(args.contexto, todos_contextos[args.contexto])
    elif args.acao == 'deletar':
        deletar_contexto(args.contexto)