import os
import json
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Tenta importar llama_cpp
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# Carrega o .env para ter acesso às chaves de API
load_dotenv()

# (A função 'configurar_servicos_interativamente' que definimos anteriormente entra aqui, sem alterações)
def configurar_servicos_interativamente():
    # ... (código completo da função da resposta anterior) ...
    config_file_path = "config_modelo_local.json"
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ ERRO: Não foi possível ler o arquivo '{config_file_path}'. Verifique o arquivo. Erro: {e}")
        return False
    servicos_a_configurar = ["sumarizador", "gerador_principal"]
    for service_name in servicos_a_configurar:
        print(f"\n--- Configurando o serviço: '{service_name}' ---")
        print(f"  Tipo atual: {config_data['servicos'][service_name].get('tipo', 'não definido')}")
        print("  Escolha o novo tipo:")
        print("    1. Modelo Local (.gguf)")
        print("    2. Modelo na Nuvem (via OpenRouter)")
        tipo_escolhido = ""
        while True:
            choice = input("  Digite sua escolha (1 ou 2): ")
            if choice == "1":
                tipo_escolhido = "local"
                break
            elif choice == "2":
                if not os.getenv("OPENROUTER_API_KEY"):
                    print("    ❌ ERRO: OPENROUTER_API_KEY não encontrada. Configure-a no .env para usar a nuvem.")
                    continue
                tipo_escolhido = "nuvem"
                break
            else:
                print("    Escolha inválida.")
        config_data['servicos'][service_name]['tipo'] = tipo_escolhido
        if tipo_escolhido == "local":
            models_dir = os.path.expanduser("~/.cache/instructlab/models/")
            if not os.path.isdir(models_dir):
                print(f"    ❌ ERRO: Diretório de modelos não encontrado em '{models_dir}'.")
                return False
            gguf_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".gguf")])
            if not gguf_files:
                print(f"    ❌ ERRO: Nenhum modelo .gguf encontrado em '{models_dir}'.")
                return False
            print("\n    -> Selecione o modelo .gguf para este serviço:")
            for i, model_name in enumerate(gguf_files):
                print(f"      {i + 1}. {model_name}")
            while True:
                try:
                    choice_model = int(input("\n    Digite o número do modelo: "))
                    if 1 <= choice_model <= len(gguf_files):
                        selected_model_path = os.path.join(models_dir, gguf_files[choice_model - 1])
                        config_data['servicos'][service_name]['path_gguf'] = selected_model_path
                        print(f"    -> Serviço '{service_name}' usará o modelo local: {gguf_files[choice_model - 1]}")
                        break
                    else:
                        print("       Escolha inválida.")
                except ValueError:
                    print("       Entrada inválida. Digite um número.")
        elif tipo_escolhido == "nuvem":
            default_id = config_data['servicos'][service_name].get('id_openrouter', '')
            prompt_text = f"\n    -> Digite o ID do modelo OpenRouter para '{service_name}' (padrão: {default_id}): "
            cloud_model_id = input(prompt_text)
            if not cloud_model_id:
                cloud_model_id = default_id
            config_data['servicos'][service_name]['id_openrouter'] = cloud_model_id
            print(f"    -> Serviço '{service_name}' usará o modelo da nuvem: {cloud_model_id}")
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2)
    print(f"\n✅ Configuração salva com sucesso em '{config_file_path}'.")
    return True

# --- INICIALIZAÇÃO DO SERVIDOR ---
if not configurar_servicos_interativamente():
    exit()

class PromptRequest(BaseModel): prompt: str

print("\n-> Iniciando o Servidor Gateway Orientado a Serviços...")
with open("config_modelo_local.json", 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
loaded_local_models = {}

if Llama:
    for service_name, service_config in CONFIG.get("servicos", {}).items():
        if service_config.get("tipo") == "local":
            model_path = service_config.get("path_gguf")
            if model_path and os.path.exists(model_path):
                print(f"-> Carregando modelo local para o serviço '{service_name}': {os.path.basename(model_path)}")
                params = CONFIG.get("parametros_carregamento_local", {})
                loaded_local_models[service_name] = Llama(model_path=model_path, **params, verbose=False)
                print(f"✅ Modelo para '{service_name}' carregado.")
            else:
                print(f"⚠️ AVISO: Modelo local para o serviço '{service_name}' não encontrado em '{model_path}'.")
else:
    print("⚠️ AVISO: 'llama-cpp-python' não está instalado. Nenhum serviço local pode ser ativado.")

app = FastAPI()

async def handle_request(service_name: str, prompt: str):
    service_config = CONFIG.get("servicos", {}).get(service_name)
    if not service_config:
        raise HTTPException(status_code=404, detail=f"Serviço '{service_name}' não encontrado na configuração.")
    service_type = service_config.get("tipo")
    params_inferencia = CONFIG.get("parametros_inferencia_padrao", {})
    
    if service_type == "local":
        model_obj = loaded_local_models.get(service_name)
        if not model_obj:
            raise HTTPException(status_code=503, detail=f"Modelo local para o serviço '{service_name}' não está carregado.")
        
        def blocking_call():
            return model_obj(prompt, stop=["[/INST]", "</s>"], **params_inferencia)
        
        # --- BLOCO CORRIGIDO ---
        try:
            # Reintroduzimos o asyncio.wait_for para garantir o timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(blocking_call),
                timeout=180.0
            )
            return {"texto_gerado": response['choices'][0]['text'].strip()}
        except asyncio.TimeoutError:
            # Capturamos o erro de timeout e retornamos um HTTP 408
            raise HTTPException(status_code=408, detail=f"A geração do serviço local '{service_name}' excedeu o limite de 180 segundos.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro no serviço local '{service_name}': {e}")
        # --- FIM DO BLOCO CORRIGIDO ---

    elif service_type == "nuvem":
        if not OPENROUTER_KEY:
            raise HTTPException(status_code=503, detail="A chave OPENROUTER_API_KEY é necessária para serviços de nuvem.")
            
        model_id = service_config.get("id_openrouter")
        headers = {"Authorization": f"Bearer {OPENROUTER_KEY}"}
        json_data = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            **params_inferencia
        }
        try:
            async with httpx.AsyncClient() as client:
                print(f"\n-> Roteando requisição do serviço '{service_name}' para OpenRouter (Modelo: {model_id})...")
                response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=json_data, timeout=180)
                response.raise_for_status()
            
            return {"texto_gerado": response.json()['choices'][0]['message']['content'].strip()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro na chamada do serviço de nuvem '{service_name}': {e}")
    else:
        raise HTTPException(status_code=501, detail=f"Tipo de serviço '{service_type}' não implementado.")


@app.post("/sumarizar")
async def endpoint_sumarizar(request: PromptRequest):
    return await handle_request("sumarizador", request.prompt)


@app.post("/gerar")
async def endpoint_gerar(request: PromptRequest):
    return await handle_request("gerador_principal", request.prompt)