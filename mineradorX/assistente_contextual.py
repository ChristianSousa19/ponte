import os
import json
import requests
from dotenv import load_dotenv

# Dependências do LangChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# (Não precisamos mais de ChatOpenAI ou da classe LLMRemotoLocal aqui!)

# --- FUNÇÃO HELPER PARA COMUNICAÇÃO COM O SERVIDOR ---

def chamar_servidor_gateway(endpoint: str, prompt_text: str) -> str:
    """
    Função centralizada para chamar um endpoint específico no nosso servidor gateway.
    'endpoint' pode ser 'sumarizar' ou 'gerar'.
    """
    try:
        # Monta a URL completa para o endpoint desejado
        url = f"http://127.0.0.1:8000/{endpoint.strip('/')}"
        
        # Envia a requisição com o prompt no corpo JSON
        response = requests.post(url, json={"prompt": prompt_text}, timeout=300)
        
        # Levanta um erro para respostas com status 4xx ou 5xx
        response.raise_for_status()
        
        # Retorna o texto gerado que vem na resposta do servidor
        return response.json().get("texto_gerado", f"ERRO: Resposta inválida do endpoint /{endpoint}")

    except requests.exceptions.Timeout:
        return f"ERRO: A requisição para o endpoint /{endpoint} excedeu o tempo limite."
    except requests.exceptions.RequestException as e:
        return f"ERRO DE CONEXÃO com o endpoint /{endpoint}: {e}"
    except Exception as e:
        return f"ERRO inesperado ao chamar o gateway: {e}"

# --- CONFIGURAÇÃO E CARREGAMENTO INICIAL ---

print("-> Configurando o ambiente do assistente (Cliente Orquestrador)...")
load_dotenv()

# Carrega o nome do modelo ATIVO a partir do JSON (que o servidor configura)
nome_modelo_principal_ativo = "Modelo Principal Desconhecido"
nome_modelo_sumarizador_ativo = "Modelo Sumarizador Desconhecido"
try:
    with open("config_modelo_local.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
        
        # Lê a configuração do serviço gerador principal
        servico_principal = config.get("servicos", {}).get("gerador_principal", {})
        if servico_principal.get("tipo") == "local":
            nome_modelo_principal_ativo = os.path.basename(servico_principal.get("path_gguf", "N/A"))
        elif servico_principal.get("tipo") == "nuvem":
            nome_modelo_principal_ativo = servico_principal.get("id_openrouter", "N/A")
            
        # Lê a configuração do serviço sumarizador
        servico_sumarizador = config.get("servicos", {}).get("sumarizador", {})
        if servico_sumarizador.get("tipo") == "local":
             nome_modelo_sumarizador_ativo = os.path.basename(servico_sumarizador.get("path_gguf", "N/A"))
        elif servico_sumarizador.get("tipo") == "nuvem":
            nome_modelo_sumarizador_ativo = servico_sumarizador.get("id_openrouter", "N/A")

except Exception as e:
    print(f"⚠️ AVISO: Não foi possível ler os nomes dos modelos da configuração: {e}")


try:
    with open("contexts.json", 'r', encoding='utf-8') as f:
        CONTEXTOS_DISPONIVEIS = json.load(f)
    with open("prompts.json", 'r', encoding='utf-8') as f:
        PROMPTS_CONFIG = json.load(f)
except FileNotFoundError as e:
    print(f"ERRO: Arquivo de configuração RAG não encontrado: {e.filename}")
    exit()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
PASTA_BASE_INDICES = "indices_rag"
print("✅ Ambiente do cliente configurado.")
print(f"   -> Modelo de Geração Principal Ativo no Servidor: {nome_modelo_principal_ativo}")
print(f"   -> Modelo de Sumarização Ativo no Servidor: {nome_modelo_sumarizador_ativo}")


# --- LÓGICA DE CHAT (ORQUESTRAÇÃO) ---

def loop_chat_rag(db: FAISS, nome_especialista: str, usar_resumo: bool):
    """
    Orquestra o fluxo de RAG, chamando os endpoints do servidor gateway
    conforme necessário.
    """
    print(f"\n✅ Especialista '{nome_especialista}' pronto!")
    print("   Digite 'sair' a qualquer momento para terminar.")
    
    while True:
        pergunta = input(f"\n🤖 Você pergunta para '{nome_especialista}': ")
        if pergunta.strip().lower() == 'sair': break

        print(f"   -> Fase 1: Buscando documentos relevantes...")
        docs_relevantes = db.similarity_search(pergunta, k=15)
        
        if not docs_relevantes:
            print("\n💡 Resposta do Especialista:\nNão encontrei documentos relevantes para esta pergunta.")
            print("-" * 20)
            continue

        contexto_original = "\n\n".join([doc.page_content for doc in docs_relevantes])
        contexto_para_geracao = contexto_original
        
        # --- ORQUESTRAÇÃO DOS ENDPOINTS ---
        if usar_resumo:
            print("   -> Fase 2a: Formatando prompt de sumarização e chamando endpoint /sumarizar...")
            prompt_sumarizacao = PROMPTS_CONFIG["sumarizacao_local"]["template"].format(
                pergunta=pergunta, contexto_completo=contexto_original
            )
            contexto_para_geracao = chamar_servidor_gateway("sumarizar", prompt_sumarizacao)
            print("   -> Contexto sumarizado recebido do servidor.")

        print("   -> Fase 2b: Formatando prompt final e chamando endpoint /gerar...")
        # Usamos um template padrão, que pode ser o 'local' ou 'nuvem' dependendo do seu gosto.
        # A lógica é a mesma.
        prompt_geracao = PROMPTS_CONFIG["geracao_rag_local"]["template"].format(
            contexto=contexto_para_geracao, pergunta=pergunta
        )
        resposta_final = chamar_servidor_gateway("gerar", prompt_geracao)
        
        print("\n💡 Resposta do Especialista:")
        print(resposta_final)
        print("-" * 20)

def loop_chat_puro():
    """
    Inicia um loop de chat geral, que sempre chama o endpoint /gerar
    do servidor gateway.
    """
    print(f"\n✅ Chat direto com o modelo principal ativo no servidor: '{nome_modelo_principal_ativo}'")
    print("   Digite 'sair' a qualquer momento para terminar.")
    
    template_string = "Você é um assistente de IA prestativo. Responda à pergunta do usuário.\n\nPERGUNTA: {pergunta}\n\nRESPOSTA:"

    while True:
        pergunta = input("\n🤖 Você pergunta: ")
        if pergunta.strip().lower() == 'sair':
            break
        
        print("   ...pensando (via servidor gateway)...")
        prompt_final = template_string.format(pergunta=pergunta)
        
        # A chamada é sempre para o endpoint de geração
        response = chamar_servidor_gateway("gerar", prompt_final)

        print("\n💡 Resposta do Gateway:")
        print(response)
        print("-" * 20)


# --- EXECUÇÃO PRINCIPAL (SIMPLIFICADA) ---

if __name__ == "__main__":
    print("\n--- Assistente de IA com Servidor Gateway ---")
    print("Escolha o modo de operação:")
    print(f"  1. Conversa Geral (com o modelo principal: {nome_modelo_principal_ativo})")
    
    opcoes_rag = {}
    # Começa o menu RAG a partir do número 2
    for i, (id_ctx, definicao_ctx) in enumerate(CONTEXTOS_DISPONIVEIS.items(), start=2):
        status = "✅ Indexado" if os.path.exists(os.path.join(PASTA_BASE_INDICES, id_ctx)) else "❌ Não Indexado"
        opcoes_rag[str(i)] = {"id": id_ctx, "nome": definicao_ctx['nome_exibicao'], "status": status}
        print(f"  {i}. Especialista RAG: {definicao_ctx['nome_exibicao']} ({status})")

    escolha_principal = input("\nDigite o número da sua opção: ")

    if escolha_principal == '1':
        loop_chat_puro()
    elif escolha_principal in opcoes_rag:
        ctx_info = opcoes_rag[escolha_principal]
        if "❌" in ctx_info["status"]:
            print(f"\nERRO: O especialista '{ctx_info['nome']}' não foi indexado.")
            exit()
        
        # Não perguntamos mais qual motor usar, a decisão está no servidor!
        usar_resumo = input("Deseja SUMARIZAR o contexto antes de enviar? (s/n, padrão 'n'): ").lower() == 's'
        
        print(f"\n-> Carregando o conhecimento do '{ctx_info['nome']}'...")
        db_contexto = FAISS.load_local(os.path.join(PASTA_BASE_INDICES, ctx_info["id"]), embeddings, allow_dangerous_deserialization=True)
        
        loop_chat_rag(db_contexto, ctx_info["nome"], usar_resumo)
    else:
        print("Escolha inválida.")
        
    print("\n👋 Sessão encerrada. Até logo!")