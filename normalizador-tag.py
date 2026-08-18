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
3. Pré normalização dos dados
4. Divisão treino/teste.
5. Treinamento dos taggers.
6. Avaliação dos modelos.
7. Aplicação dos modelos aos textos originais.
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
# 3. PRÉ-PROCESSAMENTO
# ==================================================


# ==================================================
# 4. PREPARAÇÃO DOS DADOS DE TREINAMENTO
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

# Função para divisão dos dados em train_data e test_data 
# Contendo itens dos dois corpora
# Serão usadas partições diferentes para avaliar os resultados obtidos

def dividir_dados(tuplas_transc, tuplas_tweet, train_ratio):
    split_transc = int(len(tuplas_transc) * train_ratio)
    split_tweet = int(len(tuplas_tweet) * train_ratio)

    train_data = (
        tuplas_transc[:split_transc] +
        tuplas_tweet[:split_tweet]
    )

    test_data = (
        tuplas_transc[split_transc:] +
        tuplas_tweet[split_tweet:]
    )

    # Remove listas vazias
    train_data = [x for x in train_data if x]
    test_data = [x for x in test_data if x]

    #return train_data, test_data
    return {"train": train_data,
            "test": test_data
           }

# Partições utilizadas: 70/30, 80/20 e 90/10
# Para treino/teste
splits = [0.7, 0.8, 0.9]
data = []
for s in splits:
    data.append(dividir_dados(tuplas_transc, tuplas_tweet, s))

'''
train_data1, test_data1 = dividir_dados(tuplas_transc, tuplas_tweet, 0.7)
train_data2, test_data2 = dividir_dados(tuplas_transc, tuplas_tweet, 0.8)
train_data3, test_data3 = dividir_dados(tuplas_transc, tuplas_tweet, 0.9)
'''

# Os corpora de transcrição e tweets são combinados
# para formar um único conjunto de treinamento.

# ==================================================
# 5. TREINAMENTO DOS TAGGERS
# ==================================================

# Criação de padrões para uso do tagger com regex
# O padrão final abrange todas os tokens restantes
# e atribui uma tag vazia para eles como backoff
# Assim, palavras não vistas durante o treinamento
# permanecem inalteradas na etapa de normalização.

patterns = [
     (r"([A-Za-zÀ-ÖØ-öø-ÿ])\1+", r"\1"),   # letras repetidas
     (r"([!?.,])\1+", r"\1"),              # pontuação repetida
     (r'SP$', 'São Paulo'),                # SP = são paulo quando maiúsculo
     (r"(.*)di$", "\1de"),                 # ondi - onde
     (r"(.*)ti$", "\1te"),                 # chocolati - chocolate
     (r"(.*)su$", "\1so"),                 # possu - posso
     (r"(.*)ô$", "\1ou"),                  # verbos terminados em ar
     (r"(.*)du$", "\1do"),                 # passandu - passando
     (r".*", "")                           # vazio (default)
]

# Função para treinar taggers com as partições diferentes de train_data e test_data

def treinar_taggers(train_data, patterns):
    regexp_tagger = nltk.RegexpTagger(patterns)
    uni_tagger = nltk.UnigramTagger(train_data, backoff=regexp_tagger)
    bi_tagger = nltk.BigramTagger(train_data, backoff=uni_tagger)
    tri_tagger = nltk.TrigramTagger(train_data, backoff=bi_tagger)

    return (
        regexp_tagger,
        uni_tagger,
        bi_tagger,
        tri_tagger
    )

taggers = []
for d in data:
    taggers.append(treinar_taggers(d["train"], patterns))
    
#print(regexp_tagger.tag(["noooossa"]))
# teste


'''
regexp_tagger1, uni_tagger1, bi_tagger1, tri_tagger1 = treinar_taggers(
    train_data1,
    patterns
)

regexp_tagger2, uni_tagger2, bi_tagger2, tri_tagger2 = treinar_taggers(
    train_data2,
    patterns
)

regexp_tagger3, uni_tagger3, bi_tagger3, tri_tagger3 = treinar_taggers(
    train_data3,
    patterns
)
'''


# ==================================================
# 6. AVALIAÇÃO
# ==================================================

def avaliar_taggers(trained_taggers, test_data):
    regexp_tagger, uni_tagger, bi_tagger, tri_tagger = trained_taggers
    test_regex = round(regexp_tagger.accuracy(test_data)*100, 2)
    test_uni = round(uni_tagger.accuracy(test_data)*100, 2)
    test_bi = round(bi_tagger.accuracy(test_data)*100, 2)
    test_tri = round(tri_tagger.accuracy(test_data)*100, 2)

    return(
        test_regex,
        test_uni,
        test_bi,
        test_tri
    )

tests = []
for i,d in enumerate(data):
    tests.append(avaliar_taggers(taggers[i], d["test"]))

'''
test_regex1, test_uni1, test_bi1, test_tri1 = avaliar_taggers(
    test_data1
)

test_regex2, test_uni2, test_bi2, test_tri2 = avaliar_taggers(
    test_data2
)

test_regex3, test_uni3, test_bi3, test_tri3 = avaliar_taggers(
    test_data3
)
'''

for i,s in enumerate(splits):
    print("PERFORMANCE DOS TAGGERS COM PARTIÇÃO {}/{}:".format(int(s*100), int(100-(s*100))))
    print("Regexp Tagger: {}%".format(tests[i][0]),
          "\nUnigram Tagger: {}%".format(tests[i][1]),
          "\nBigram Tagger: {}%".format(tests[i][2]),
          "\nTrigram Tagger: {}%".format(tests[i][3])
         )

#Encontrar melhor performance

nomes = ["Regexp", "Unigram", "Bigram", "Trigram"]

melhor_acuracia = 0
melhor_particao = 0      
melhor_tagger = 0        

for i, resultado in enumerate(tests):
    for j, acuracia in enumerate(resultado):
        if acuracia > melhor_acuracia:
            melhor_acuracia = acuracia
            melhor_particao = i
            melhor_tagger = j

print(
    f"Melhor resultado: {nomes[melhor_tagger]} "
    f"({int(splits[melhor_particao]*100)}/{int(100-(splits[melhor_particao]*100))}) "
    f"- {melhor_acuracia:.2f}%"
)
melhor_modelo = taggers[melhor_particao][melhor_tagger]

# ==================================================
# 7. NORMALIZAÇÃO DOS TEXTOS
# ==================================================

# função para normalizar os textos originais por sentença 
# aplicação do Tagger com melhor performance

def normalizar(texto):
    resultado = [] # guarda sentença normalizada
    soma = 0 # quantidade de tokens normalizados
    exemplos = [] # alguns tokens normalizados

    for sent in texto:
        #tokenização das sentenças
        tokens = re.findall(TOKEN_REGEX, sent)
        normalizados = []
        
        for token, normalizacao in melhor_modelo.tag(tokens):
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
    print("Exemplos de normalizações:")
    
    for original, normalizado in exemplos:
        print(f"{original} -> {normalizado}")

    return resultado

# normalização dos corpora
print("NORMALIZAÇÕES USANDO O TAGGER COM MELHOR PERFORMANCE")
print("TRANSCRIÇÃO DE DIÁLOGO ADULTO-CRIANÇA")
print("Texto original:")
print(transcricao_da[:20])
transcricao_normalizada = normalizar(transcricao_da)
print("Texto normalizado:")
print(transcricao_normalizada [:20])

print("POSTAGENS NO X (TWITTER)")
print("Texto original:")
print(raw_tweets[12:19])
tweets_normalizados = normalizar(raw_tweets)
print("Texto normalizado:")
print(tweets_normalizados [12:19])