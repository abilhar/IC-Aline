# ==================================================
# 1. IMPORTAÇÃO DE BIBLIOTECAS
# ==================================================
import re
import nltk 

"""
Pipeline de normalização com tagger.

Etapas:
1. Carregamento dos corpora originais e anotados.
2. Conversão das anotações em tuplas,
   formato de leitura dos taggers (original, normalização).
3. Divisão treino/teste.
4. Treinamento dos taggers.
5. Avaliação dos modelos.
6. Aplicação dos modelos aos textos originais.
"""

# ==================================================
# 2. CARREGAMENTO DOS CORPORA
# ==================================================
# Dados originais
with open("tweets-teste-2000.csv", "r", encoding="utf-8") as tweets:
    raw_tweets = tweets.read().splitlines()

with open("transc-da-teste.txt", "r", encoding="utf-8") as transc:
    transcricao_da = transc.read().splitlines()

# remoção de cabeçalho dos dados originais
transcricao_da = transcricao_da[16:]
raw_tweets = raw_tweets[1:]

# Dados anotados para treinamento e teste
# tweets e transcrição de conversa adulto/criança 
# com anotações nas palavras a serem normalizadas - ex: "vc--/você"

with open("2000tweets-2705.csv", "r", encoding="utf-8") as tweets:
    tweets_controle = tweets.read().splitlines()

with open("transc_da_anotada-2705.txt", "r", encoding="utf-8") as transcricao:
    transcricao_controle = transcricao.read().splitlines()

# remoção de cabeçalho nos dados de controle
tweets_controle = tweets_controle[1:]
transcricao_controle = transcricao_controle[16:]

# remoção de links dos tweets
raw_tweets = [re.sub(r"https?:\/\/\S+", "", item) for item in raw_tweets]
tweets_controle = [re.sub(r"https?:\/\/\S+", "", item) for item in tweets_controle]

#lower case
raw_tweets = [line.lower() for line in raw_tweets]
transcricao_da = [line.lower() for line in transcricao_da]
tweets_controle = [line.lower() for line in tweets_controle]
transcricao_controle = [line.lower() for line in transcricao_controle]

'''
Regex geral criado para tokenizar corretamente palavras características
dos tipos de corpora usados (textos digitais e transcrição de fala).
Evita separar palavras que possuem caracteres além de letras em mais de um token.
'''

TOKEN_REGEX = (
    r"[:;=][()DdPp]+"                                    #emoticons comuns que não devem ser separados
    r"|[A-Za-zÀ-ÖØ-öø-ÿ_]+--/[A-Za-zÀ-ÖØ-öø-ÿ_]+"        #anotações
    r"|p\.:"
    r"|d\.:"
    r"|a\.:"                                              #letras de identificação de falantes
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"    #emails
    r"|@[A-Za-z0-9_]+"                                    #nomes de usuário
    r"|#[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+"                           #hashtags
    r"|[A-Za-zÀ-ÖØ-öø-ÿ_]+"                               #outras palavras
    r"|."                                                 #outras pontuações
)

# ==================================================
# 3. PREPARAÇÃO DOS DADOS DE TREINAMENTO
# ==================================================

'''
Criação de tuplas a serem utilizadas no treinamento dos taggers.
A função converte uma sentença anotada em uma lista de tuplas.

    Exemplo:
        "Tenho raiva de vc--/você por vc--/você ser tão linda"" -->
        "('Tenho', ''), ('raiva', ''), ('de', ''), ('vc', 'você'),
        ('por', ''), ('vc', 'você'), ('ser', ''), ('tão', ''), ('linda', '')"

A estrutura gerada é compatível com os taggers do NLTK,
onde o primeiro elemento corresponde ao token original
e o segundo à sua forma normalizada (se houver anotação).
'''

def criar_tuplas(texto):
    #tokenizar sentenças
    tokens = [t for t in re.findall(TOKEN_REGEX, texto) if not t.isspace()]

    #lista de tuplas prontas
    resultado = []

    for token in tokens:
        if "--/" in token: #se houver, separar pelo padrão de anotação "--/"
            original, substituicao = token.split("--/", 1)
            resultado.append((original, substituicao))

        else: #se não houver anotação, deixar segundo item da tupla vazio
            resultado.append((token, ""))

    return resultado

# usar função criar_tuplas nos corpora anotados (tweets e transcrição)
# para serem utilizados no treinamento dos taggers

tuplas_transc = [criar_tuplas(item) for item in transcricao_controle]
tuplas_tweet = [criar_tuplas(item) for item in tweets_controle]

# soma de tokens
print("TOTAL DE TOKENS CARREGADOS PARA O TREINAMENTO:")
tokens_transc = sum(len(sent) for sent in tuplas_transc)
tokens_tweet = sum(len(sent) for sent in tuplas_tweet)
print("transcrição:", tokens_transc)
print("tweets:", tokens_tweet)

# Separação dos dados anotados em conjuntos de treinamento (80%)
# e avaliação (20%) para medir o desempenho dos taggers.
TRAIN_RATIO = 0.8
split_transc = int(len(tuplas_transc) * TRAIN_RATIO)
split_tweet = int(len(tuplas_tweet) * TRAIN_RATIO)

train_data = (
    tuplas_transc[:split_transc]
    + tuplas_tweet[:split_tweet]
)
train_data = [x for x in train_data if x] #remoção de listas vazias

test_data = (
    tuplas_transc[split_transc:]
    + tuplas_tweet[split_tweet:]
)
test_data = [x for x in test_data if x] #remoção de listas vazias

# Os corpora de transcrição e tweets são combinados
# para formar um único conjunto de treinamento.

# ==================================================
# 4. TREINAMENTO DOS TAGGERS
# ==================================================

# O tagger padrão retorna uma string vazia para qualquer token sem anotação.
# Assim, palavras não vistas durante o treinamento
# permanecem inalteradas na etapa de normalização.

default = nltk.DefaultTagger('')
uni_tagger = nltk.UnigramTagger(train_data, backoff=default)
bi_tagger = nltk.BigramTagger(train_data, backoff=default)

# ==================================================
# 5. AVALIAÇÃO
# ==================================================

print("PERFORMANCE DOS TAGGERS TREINADOS")
print("Unigram =", round(uni_tagger.accuracy(test_data)*100, 2), "%")
print("Bigram =", round(bi_tagger.accuracy(test_data)*100, 2), "%")

# ==================================================
# 6. NORMALIZAÇÃO DOS TEXTOS
# ==================================================

# função para normalizar os textos originais por sentença 
# aplicação do UnigramTagger aos textos originais

def normalizar_unigrama(texto):
    resultado = [] # guarda sentença normalizada
    soma = 0 # quantidade de tokens normalizados
    exemplos = [] # alguns tokens normalizados

    for sent in texto:
        #tokenização das sentenças
        tokens = re.findall(TOKEN_REGEX, sent)
        normalizados = []
        
        for token, normalizacao in uni_tagger.tag(tokens):
            if normalizacao:
                # mantém apenas normalização se houver
                normalizados.append(normalizacao)
                # soma quantos tokens foram normalizados
                soma += 1
                # criação de lista de exemplos
                if len(exemplos) < 10 and (token, normalizacao) not in exemplos:
                    exemplos.append((token, normalizacao))
                    
            else:
                # se não houver normalização, mantém o token original
                normalizados.append(token)

        # recompõe a sentença mantendo a tokenização original
        resultado.append(''.join(normalizados))
        
    print("Total de tokens normalizados:", soma)
    print("Exemplos de tokens e normalizações:")
    
    for original, normalizado in exemplos:
        print(f"{original} -> {normalizado}")

    return resultado

# normalização dos corpora
print("NORMALIZAÇÃO DE TRANSCRIÇÃO DE DIÁLOGO ADULTO-CRIANÇA")
print("Texto original:")
print(transcricao_da[:20])
transcricao_normalizada_unigrama = normalizar_unigrama(transcricao_da)
print("Texto normalizado:")
print(transcricao_normalizada_unigrama [:20])

print("NORMALIZAÇÃO DE TWEETS")
print("Texto original:")
print(raw_tweets[12:19])
tweets_normalizados_unigrama = normalizar_unigrama(raw_tweets)
print("Texto normalizado:")
print(tweets_normalizados_unigrama [12:19])
