import tweepy
import pandas as pd
from datetime import datetime, timedelta
import time # Importamos a biblioteca 'time' para usar a função de espera
from dotenv import load_dotenv
import os

# --- CARREGAR VARIÁVEIS DO .env ---
load_dotenv()  # Carrega as variáveis do arquivo .env

# --- CONFIGURAÇÃO DO BEARER TOKEN ---
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
if not BEARER_TOKEN:
    raise ValueError("A variável de ambiente BEARER_TOKEN não está definida.")

MAX_DAYS_AGO = 7
RETRY_PAUSE_MINUTES = 5
MAX_RETRIES = 3

def build_query_part(items_str, prefix="", operator="OR"):
    """Função auxiliar para construir parte da query para múltiplos itens."""
    if not items_str:
        return ""
    
    items = [item.strip() for item in items_str.split(',')]
    formatted_items = [f'"{item}"' if ' ' in item else item for item in items]
    prefixed_items = [f"{prefix}{item}" for item in formatted_items]
    
    if len(prefixed_items) > 1:
        return f"({f' {operator} '.join(prefixed_items)})"
    else:
        return prefixed_items[0]

def coletar_dados_api():
    """
    Função principal para coletar dados do X (Twitter) usando a API v2 com filtros avançados.
    """
    print("--- Assistente de Coleta de Dados do X v12 (API v2 com Texto Completo) ---")
    
    if "SEU_BEARER_TOKEN_VEM_AQUI" in BEARER_TOKEN or not BEARER_TOKEN:
        print("❌ ERRO: O Bearer Token não foi definido no script.")
        print("Abra o arquivo e insira seu token na variável BEARER_TOKEN.")
        return

    try:
        client = tweepy.Client(BEARER_TOKEN)
        print("✅ Cliente da API conectado com sucesso usando o token fixo.")
    except Exception as e:
        print(f"❌ ERRO: Falha ao inicializar o cliente da API. Erro: {e}")
        return

    print("\n--- Filtros de Inclusão (separe múltiplos itens com vírgula) ---")
    users_str = input("1. Incluir usuários? (ex: iFood,nubank): ").strip()
    hashtags_str = input("2. Incluir hashtags? (ex: IA,fintechs): ").strip()
    text_phrase_str = input("3. Incluir texto ou frases? (ex: compra de mercado,nova função): ").strip()

    print("\n--- Filtros Opcionais ---")
    days_ago_str = input(f"4. Pesquisar nos últimos X dias? (máximo de {MAX_DAYS_AGO}): ").strip()
    language_str = input("5. Filtrar por idioma(s)? (ex: pt,en,es): ").strip()

    print("\n--- Filtros de Exclusão (separe múltiplos itens com vírgula) ---")
    exclude_str = input("6. Excluir palavras ou frases? (ex: promoção,vale a pena): ").strip()

    try:
        limit = int(input("\nQual o número máximo de tweets a coletar? (ex: 50): ").strip())
    except ValueError:
        print("Limite inválido. Usando o padrão de 50.")
        limit = 50

    # --- Construção da Query Dinâmica Avançada ---
    query_parts = [
        build_query_part(users_str, prefix="from:"),
        build_query_part(hashtags_str, prefix="#"),
        build_query_part(text_phrase_str),
        build_query_part(language_str, prefix="lang:")
    ]
    
    if exclude_str:
        exclude_items = []
        for item in exclude_str.split(','):
            item = item.strip()
            if ' ' in item:
                exclude_items.append(f'-"{item}"')
            else:
                exclude_items.append(f'-{item}')
        query_parts.extend(exclude_items)

    query_parts = [part for part in query_parts if part]

    if not any(p for p in query_parts if not p.startswith('-')):
        print("\n❌ ERRO: Nenhum filtro de inclusão foi fornecido.")
        return
        
    final_query = " ".join(query_parts)
    print(f"\n🔍 Construindo a query: {final_query}")
    print(f"Buscando até {limit} tweets...")
    
    start_time = None
    if days_ago_str and days_ago_str.isdigit():
        days_ago = min(int(days_ago_str), MAX_DAYS_AGO)
        if int(days_ago_str) > MAX_DAYS_AGO:
            print(f"⚠️ AVISO: O limite de busca é de {MAX_DAYS_AGO} dias. Ajustando a busca.")
        start_time = datetime.utcnow() - timedelta(days=days_ago)
        print(f"Filtrando tweets desde: {start_time.strftime('%Y-%m-%d')}")

    # --- Coleta de Dados ---
    tweets_list = []
    tentativas = 0
    sucesso = False

    while tentativas <= MAX_RETRIES and not sucesso:
        try:
            # --- MELHORIA: PEDINDO O TEXTO COMPLETO ---
            # Adicionamos 'referenced_tweets.id' para pegar o tweet original de um RT
            response = client.search_recent_tweets(
                query=final_query, 
                max_results=min(limit, 100), 
                tweet_fields=["created_at", "author_id", "lang", "referenced_tweets"],
                start_time=start_time,
                expansions=["author_id", "referenced_tweets.id"]
            )
            
            sucesso = True
            
            # --- MELHORIA: CRIANDO UM "DICIONÁRIO" DE TWEETS ORIGINAIS ---
            # Guardamos todos os tweets incluídos (os originais dos RTs) para consulta rápida
            included_tweets = {tweet.id: tweet for tweet in response.includes.get('tweets', [])}
            users = {user.id: user.username for user in response.includes.get('users', [])}

            if response.data:
                for tweet in response.data:
                    author_username = users.get(tweet.author_id, "usuário_desconhecido")
                    
                    # --- MELHORIA: LÓGICA PARA PEGAR O TEXTO COMPLETO ---
                    final_text = tweet.text
                    # Verifica se é um Retweet
                    if tweet.referenced_tweets:
                        for ref_tweet in tweet.referenced_tweets:
                            if ref_tweet.type == 'retweeted':
                                # Se for RT, busca o texto do tweet original no nosso dicionário
                                original_tweet = included_tweets.get(ref_tweet.id)
                                if original_tweet:
                                    final_text = original_tweet.text
                    
                    tweets_list.append([tweet.created_at, author_username, final_text, tweet.lang])
            else:
                print("\nNenhum tweet encontrado para os filtros fornecidos.")
                return

        except tweepy.errors.TooManyRequests:
            tentativas += 1
            if tentativas > MAX_RETRIES:
                print(f"\n❌ ERRO: Número máximo de {MAX_RETRIES} tentativas atingido. Abortando.")
                break

            print(f"\n⏳ AVISO: Limite de requisições da API atingido. Tentativa {tentativas}/{MAX_RETRIES}.")
            print(f"O script fará uma pausa de {RETRY_PAUSE_MINUTES} minutos.")
            print("Pressione Ctrl+C para cancelar a espera.")
            try:
                time.sleep(RETRY_PAUSE_MINUTES * 60 + 1)
                print("\n⏰ Pausa concluída. Tentando novamente...")
            except KeyboardInterrupt:
                print("\nExecução cancelada pelo usuário durante a pausa.")
                return
        except Exception as e:
            print(f"\n❌ Ocorreu um erro inesperado durante a coleta: {e}")
            return

    # --- Processamento e Salvamento ---
    if not tweets_list:
        return

    df = pd.DataFrame(tweets_list, columns=['datetime', 'username', 'content', 'language'])
    df.drop_duplicates(subset=['content'], inplace=True)
    df.dropna(inplace=True)

    filename = f"base_conhecimento_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    print(f"\n💾 Salvando {len(df)} tweets únicos no arquivo: {filename}")

    with open(filename, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            f.write(f"### Tweet de @{row['username']} em {row['datetime'].strftime('%Y-%m-%d %H:%M')} (Idioma: {row['language']})\n")
            f.write(row['content'].replace('\n', ' ') + '\n\n')

    print("\n✅ Missão concluída! Sua base de conhecimento está agora com o texto completo.")

if __name__ == '__main__':
    coletar_dados_api()
