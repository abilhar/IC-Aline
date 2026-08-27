import re
import nltk 
import pandas as pd
'''
# ==================================================
# NORMALIZADOR AUTOMÁTICO DE TEXTOS INFORMAIS
# ==================================================

Objetivo:
Desenvolvimento e avaliação de um sistema de normalização
automática de textos informais em português brasileiro.

Corpora utilizados:
- Transcrições de interação adulto-criança
- Postagens do X (Twitter)

PIPELINE - etapas:
1. Carregamento dos corpora
2. Pré-normalização por regras linguísticas e regex
3. Preparação dos dados anotados para treinamento e teste
4. Treinamento dos modelos
5. Avaliação dos modelos
6. Seleção do melhor modelo pelo F-score
7. Normalização dos textos originais

Modelos avaliados:
- Unigram Tagger
- Bigram Tagger
- Trigram Tagger

Partições de treino/teste:
- 70/30
- 80/20
- 90/10

Métricas utilizadas:
- Acurácia
- Precisão
- Sensibilidade (Recall)
- F-score

O melhor modelo é aplicado aos textos após a pré-normalização.
'''
print("=" * 65)
print("NORMALIZADOR AUTOMÁTICO DE TEXTOS INFORMAIS")
print("=" * 65)
print("Objetivo: normalização automática de textos informais em português")
print("Corpora: transcrição de interação adulto-criança e postagens do X (Twitter)")
print("Modelos: Unigram, Bigram e Trigram Tagger")
print("Partições: 70/30, 80/20 e 90/10")
print("Métricas: Acurácia, Precisão, Sensibilidade e F-score")
print("Seleção final: modelo com maior F-score")
print("=" * 65)


# ==================================================
# 1. CARREGAMENTO DOS CORPORA
# ==================================================
# Dados originais
with open("tweets-teste-2000.csv", "r", encoding="utf-8") as tweets:
    raw_tweets = tweets.read().splitlines()

with open("transc-da-teste.txt", "r", encoding="utf-8") as transc:
    transcricao_da = transc.read().splitlines()

# Dicionário do léxico brasileiro para base
with open("lexporbr.csv", "r", encoding="utf-8") as lex:
    lexico = set(lex.read().splitlines())

# Dados anotados para treinamento e teste
# tweets e transcrição de conversa adulto/criança 
# com anotações nas palavras a serem normalizadas: "vc--/você"

with open("2000tweets-2705.csv", "r", encoding="utf-8") as tweets:
    tweets_controle = tweets.read().splitlines()

with open("transc_da_anotada-2705.txt", "r", encoding="utf-8") as transcricao:
    transcricao_controle = transcricao.read().splitlines()

# remoção de cabeçalho nos dados originais e de controle
transcricao_da = transcricao_da[16:]
raw_tweets = raw_tweets[2:]
transcricao_controle = transcricao_controle[16:]
tweets_controle = tweets_controle[1:]

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
    r"|[!?.,]+"                                           #sequências de pontuação
    r"|[^\s]"                                             #outros caracteres
)

# ==================================================
# 2. PRÉ-NORMALIZAÇÃO
# ==================================================
# etapa de pré-processamento para aumentar eficácia dos modelos
# função para exclusão de letras repetidas fora do padrão lexical do português.

def excluir_letras_repetidas(texto):

    ocorrencias = list(re.finditer(TOKEN_REGEX, texto))
    novas = []
    ultima_posicao = 0

    for ocorrencia in ocorrencias:

        # trecho entre o token anterior e o atual
        novas.append(
            texto[ultima_posicao:ocorrencia.start()]
        )

        p = ocorrencia.group()

        # mantém nomes de usuário, hashtags, e-mails e risada "kkkk"
        if (
            p.startswith("@")
            or p.startswith("#")
            or ("@" in p and "." in p)
            or re.fullmatch(r"k{2,}", p)
        ):
            novas.append(p)

        else:

            # reduz pontuação repetida
            p = re.sub(
                r"([!?.,])\1+",
                r"\1",
                p
            )

            if p.isalpha():

                # mantém palavras que já existem no léxico
                if p in lexico:
                    novas.append(p)
                    ultima_posicao = ocorrencia.end()
                    continue

                # reduz vogais repetidas para duas
                palavra_duas_vogais = re.sub(
                    r"([AEIOUaeiouÁÉÍÓÚáéíóúÀàÂâÊêÔôÃãÕõ])\1{2,}",
                    r"\1\1",
                    p
                )

                if palavra_duas_vogais in lexico:
                    novas.append(palavra_duas_vogais)
                    ultima_posicao = ocorrencia.end()
                    continue

                # reduz vogais repetidas para uma
                palavra_uma_vogal = re.sub(
                    r"([AEIOUaeiouÁÉÍÓÚáéíóúÀàÂâÊêÔôÃãÕõ])\1+",
                    r"\1",
                    palavra_duas_vogais
                )

                if palavra_uma_vogal in lexico:
                    novas.append(palavra_uma_vogal)
                    ultima_posicao = ocorrencia.end()
                    continue

                # reduz consoantes repetidas para duas
                palavra_duas_consoantes = re.sub(
                    r"([B-DF-HJ-NP-TV-Zb-df-hj-np-tv-z])\1{2,}",
                    r"\1\1",
                    palavra_uma_vogal
                )

                if palavra_duas_consoantes in lexico:
                    novas.append(palavra_duas_consoantes)
                    ultima_posicao = ocorrencia.end()
                    continue

                # reduz consoantes repetidas para uma
                palavra_uma_consoante = re.sub(
                    r"([B-DF-HJ-NP-TV-Zb-df-hj-np-tv-z])\1+",
                    r"\1",
                    palavra_duas_consoantes
                )

                novas.append(palavra_uma_consoante)

            else:
                novas.append(p)

        ultima_posicao = ocorrencia.end()

    # adiciona o restante da sentença
    novas.append(texto[ultima_posicao:])

    return "".join(novas)

# normalização por substituições utilizando padrões
# padrões com regex

patterns = [
     (r"([!?.,])\1+", r"\1"),               # pontuação repetida
     (r"(.*)di$", r"\1de"),                 # ondi - onde
     (r"(.*)ti$", r"\1te"),                 # chocolati - chocolate
     (r"(.*)su$", r"\1so"),                 # possu - posso
     (r"(.*)ô$", r"\1ou"),                  # passô - passou
     (r"(.*)du$", r"\1do"),                 # passandu - passando
     (r"(.*)á$", r"\1ar"),                  # falá - falar
     (r"(.*)ê$", r"\1er"),                  # fazê - fazer
]

def normalizar_pattern(texto):

    # encontra os tokens e suas posições no texto original
    ocorrencias = list(re.finditer(TOKEN_REGEX, texto))

    novas = []
    ultima_posicao = 0

    for ocorrencia in ocorrencias:

        # preserva espaços e outros caracteres entre os tokens
        novas.append(
            texto[ultima_posicao:ocorrencia.start()]
        )

        p = ocorrencia.group()

        # mantém nomes de usuário, hashtags, e-mails e risada "kkkk"
        if (
            p.startswith("@")
            or p.startswith("#")
            or ("@" in p and "." in p)
            or re.fullmatch(r"k{2,}", p)
        ):
            novas.append(p)
            ultima_posicao = ocorrencia.end()
            continue

        # se a palavra já existe no léxico, mantém
        if p in lexico:
            novas.append(p)
            ultima_posicao = ocorrencia.end()
            continue

        palavra_normalizada = p

        # testa os padrões
        for padrao, substituicao in patterns:

            nova_palavra = re.sub(
                padrao,
                substituicao,
                palavra_normalizada
            )

            # verifica se a substituição realmente alterou
            # a palavra
            if nova_palavra != palavra_normalizada:

                # verifica se a palavra resultante
                # existe no léxico
                if nova_palavra in lexico:
                    palavra_normalizada = nova_palavra
                    break

        novas.append(palavra_normalizada)

        ultima_posicao = ocorrencia.end()

    # adiciona o restante da sentença
    novas.append(texto[ultima_posicao:])

    return "".join(novas) 

# Pré normalização dos dados
transcricao_pre = [excluir_letras_repetidas(line) for line in transcricao_da]
tweets_pre = [excluir_letras_repetidas(line) for line in raw_tweets]

transcricao_pre = [normalizar_pattern(patterns, line) for line in transcricao_pre]
tweets_pre = [normalizar_pattern(patterns, line) for line in tweets_pre]

# ==================================================
# 3. PREPARAÇÃO DOS DADOS PARA TREINAMENTO E TESTE
# ==================================================

'''
Criação de tuplas a serem utilizadas no treinamento dos modelos.
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
# para serem utilizados no treinamento dos modelos

tuplas_transc = [criar_tuplas(item) for item in transcricao_controle]
tuplas_tweet = [criar_tuplas(item) for item in tweets_controle]

# dados base para cálculos estatísticos
# Sempre na ordem transcrição + tweets
dados_estatistica = (
    tuplas_transc +
    tuplas_tweet
)

'''
# Conferência de formato
print("Número de sentenças:", len(dados_estatistica))
print("Primeira sentença:")
print(dados_estatistica[0])

# Conferência de tokens que possuem normalização
contador = 0

for sentenca in dados_estatistica:
    for original, esperado in sentenca:
        if esperado != "":
            print(original, "->", esperado)
            contador += 1

            if contador == 20:
                break

    if contador == 20:
        break
'''
# soma de tokens
print("\n" + "=" * 50)
print("TOTAL DE PALAVRAS CARREGADAS:")
print("=" * 50)
tokens_transc = sum(len(sent) for sent in tuplas_transc)
tokens_tweet = sum(len(sent) for sent in tuplas_tweet)
print("TRANSCRIÇÃO:", tokens_transc)
print("TWEETS:", tokens_tweet)

# Função para divisão dos dados em train_data e test_data 
# Contendo itens dos dois corpora
# Serão usadas 3 partições diferentes para avaliar os resultados obtidos

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
    
# Os corpora de transcrição e tweets são combinados
# para formar um único conjunto de treinamento.

'''
print("\n" + "=" * 50)
print("CONFERÊNCIA DAS PARTIÇÕES")
print("=" * 50)

for i, s in enumerate(splits):

    print(
        f"\nPartição {int(s*100)}/{int((1-s)*100)}"
    )

    print(
        "Sentenças de treino:",
        len(data[i]["train"])
    )

    print(
        "Sentenças de teste:",
        len(data[i]["test"])
    )

    print("\nPrimeira sentença de teste:")

    print(data[i]["test"][0])
'''

# ==================================================
# 4. TREINAMENTO DOS MODELOS
# ==================================================

# Como backoff, será utilizada uma tag vazia
# Assim, palavras não vistas durante o treinamento
# permanecem inalteradas na etapa de normalização.

default = nltk.DefaultTagger('')

# Função para treinar modelos

def treinar_modelos(train_data, default):
    uni_tagger = nltk.UnigramTagger(train_data, backoff=default)
    bi_tagger = nltk.BigramTagger(train_data, backoff=uni_tagger)
    tri_tagger = nltk.TrigramTagger(train_data, backoff=bi_tagger)

    return (
        uni_tagger,
        bi_tagger,
        tri_tagger
    )

modelos = []
for d in data:
    modelos.append(treinar_modelos(d["train"], default))
'''
# TESTE DA PREVISÃO TOKEN A TOKEN

modelo = modelos[0][0]       # Unigram 70/30
test_data = data[0]["test"]

sentenca = test_data[12]

tokens = [original for original, esperado in sentenca]

previsoes = modelo.tag(tokens)

print("Esperado:")
print(sentenca)

print("\nPrevisto:")
print(previsoes)

# ==================================================
# TESTE E CONTAGEM DOS RESULTADOS
# ==================================================

def contar_resultados(test_data, modelo):

    tp = tn = fp = fn = 0

    # percorre cada sentença do conjunto de teste
    for sentenca in test_data:

        # pega somente os tokens originais
        tokens = [original for original, esperado in sentenca]

        # aplica o modelo à sentença
        previsto = modelo.tag(tokens)

        # compara token por token
        for (original, esperado), (token_previsto, normalizacao_prevista) in zip(
            sentenca,
            previsto
        ):

            # TP: deveria normalizar e normalizou corretamente
            if esperado != "" and normalizacao_prevista == esperado:
                tp += 1

            # TN: não deveria normalizar e não normalizou
            elif esperado == "" and normalizacao_prevista == "":
                tn += 1

            # FP: não deveria normalizar, mas normalizou
            elif esperado == "" and normalizacao_prevista != "":
                fp += 1

            # FN: deveria normalizar, mas não normalizou
            # ou normalizou para a forma errada
            elif esperado != "" and normalizacao_prevista != esperado:
                fn += 1

    return tp, tn, fp, fn


modelo = modelos[0][0]
test_data = data[0]["test"]

tp, tn, fp, fn = contar_resultados(test_data, modelo)

print("\n" + "=" * 50)
print("RESULTADOS")
print("=" * 50)

print("TP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)

acuracia = (tp + tn) / (tp + tn + fp + fn) * 100

print("Acurácia calculada:", acuracia)
'''
# ==================================================
# 5. AVALIAÇÃO DOS MODELOS
# ==================================================
'''
Avaliação da acurácia de cada modelo 
a partir da função do NLTK
com as 3 partições diferentes
'''
def avaliar_modelos(trained_taggers, test_data):
    uni_tagger, bi_tagger, tri_tagger = trained_taggers
    test_uni = round(uni_tagger.accuracy(test_data)*100, 2)
    test_bi = round(bi_tagger.accuracy(test_data)*100, 2)
    test_tri = round(tri_tagger.accuracy(test_data)*100, 2)

    return(
        test_uni,
        test_bi,
        test_tri
    )

tests = []
for i,d in enumerate(data):
    tests.append(avaliar_modelos(modelos[i], d["test"]))

# ==================================================
# AVALIAÇÃO ESTATÍSTICA GERAL
# ==================================================

'''
Avaliação estatística dos modelos.

A avaliação é feita diretamente sobre os tokens das
sentenças anotadas, sem reconstruir ou tokenizar novamente
os textos normalizados.

Definições:

TP (Verdadeiro Positivo):
A palavra deveria ser normalizada e o sistema normalizou
corretamente.

TN (Verdadeiro Negativo):
A palavra não deveria ser normalizada e o sistema a manteve.

FP (Falso Positivo):
A palavra não deveria ser normalizada, mas o sistema
produziu uma normalização.

FN (Falso Negativo):
A palavra deveria ser normalizada, mas o sistema não
produziu a normalização correta.

Métricas calculadas:
- Acurácia
- Precisão
- Sensibilidade
- F-score
'''
# Função para cálculo de todas as métricas
def calcular_metricas(dados, modelo):

    tp = tn = fp = fn = 0

    for sentenca in dados:

        tokens = [
            original
            for original, esperado in sentenca
        ]

        previsoes = modelo.tag(tokens)

        for (original, esperado), (token, previsto) in zip(
            sentenca,
            previsoes
        ):

            if esperado != "" and previsto == esperado:
                tp += 1

            elif esperado == "" and previsto == "":
                tn += 1

            elif esperado == "" and previsto != "":
                fp += 1

            elif esperado != "" and previsto != esperado:
                fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Acurácia": accuracy * 100,
        "Precisão": precision * 100,
        "Sensibilidade": recall * 100,
        "F-score": f1 * 100
    }

# AVALIAÇÃO DOS 9 MODELOS
nomes_modelos = ["Unigram", "Bigram", "Trigram"]

resultados = []

for i, split in enumerate(splits):

    particao = f"{int(split * 100)}/{round((1 - split) * 100)}"

    # DEFINE O CONJUNTO DE TESTE
    test_data = data[i]["test"]

    for j, nome_modelo in enumerate(nomes_modelos):

        modelo = modelos[i][j]

        #print(f"Avaliando {nome_modelo} - {particao}")

        metricas = calcular_metricas(
            test_data,
            modelo
        )

        metricas["Partição"] = particao
        metricas["Modelo"] = nome_modelo

        resultados.append(metricas)

# ==================================================
# TABELA COM RESULTADOS
# ==================================================

df_resultados = pd.DataFrame(resultados)

df_resultados = df_resultados[
    [
        "Partição",
        "Modelo",
        "TP",
        "TN",
        "FP",
        "FN",
        "Acurácia",
        "Precisão",
        "Sensibilidade",
        "F-score"
    ]
]


print("\n" + "=" * 60)
print("RESULTADOS ESTATÍSTICOS")
print("=" * 60)
print(df_resultados.round(2))

'''
# checagem de consistência da implementação dos cálculos estatísticos

print("\n" + "=" * 60)
print("CONFERÊNCIA: ACURÁCIA DA FUNÇÃO x ACURÁCIA DO NLTK")
print("=" * 60)

for i, split in enumerate(splits):

    particao = f"{int(split * 100)}/{round((1 - split) * 100)}"

    # conjunto de teste da partição
    test_data = data[i]["test"]

    for j, nome_modelo in enumerate(nomes_modelos):

        # modelo correspondente
        modelo = modelos[i][j]

        # acurácia calculada pelo NLTK
        acuracia_nltk = modelo.accuracy(test_data) * 100

        # acurácia calculada pela minha função
        linha = df_resultados[
            (df_resultados["Partição"] == particao) &
            (df_resultados["Modelo"] == nome_modelo)
        ]

        acuracia_calculada = linha["Acurácia"].iloc[0]

        # diferença entre os dois valores
        diferenca = abs(
            acuracia_calculada - acuracia_nltk
        )

        print(
            f"{particao} - {nome_modelo}: "
            f"Função = {acuracia_calculada:.6f}% | "
            f"NLTK = {acuracia_nltk:.6f}% | "
            f"Diferença = {diferenca:.10f}"
        )
'''

# ==================================================
# 6. SELEÇÃO DO MELHOR MODELO POR F-SCORE
# ==================================================

melhor_indice = df_resultados["F-score"].idxmax()

melhor_resultado = df_resultados.loc[melhor_indice]

print("\n" + "=" * 50)
print("MELHOR MODELO")
print("=" * 50)

print("Partição:", melhor_resultado["Partição"])
print("Modelo:", melhor_resultado["Modelo"])
print("F-score:", round(melhor_resultado["F-score"], 2))

# Criação de variável contento os índices do melhor modelo
particoes = {
    "70/30": 0,
    "80/20": 1,
    "90/10": 2
}

indices_modelos = {
    "Unigram": 0,
    "Bigram": 1,
    "Trigram": 2
}

indice_particao = particoes[melhor_resultado["Partição"]]
indice_modelo = indices_modelos[melhor_resultado["Modelo"]]

melhor_modelo = modelos[indice_particao][indice_modelo]

# ==================================================
# 7. NORMALIZAÇÃO DOS TEXTOS
# ==================================================

def normalizar(texto, modelo):

    resultado = []
    soma = 0
    exemplos = []

    for sentenca in texto:

        # 1. Encontrar os tokens e suas posições
        ocorrencias = list(
            re.finditer(TOKEN_REGEX, sentenca)
        )

        tokens = [
            ocorrencia.group()
            for ocorrencia in ocorrencias
        ]

        # 2. Aplicar o modelo aos tokens
        previsoes = modelo.tag(tokens)

        # 3. Reconstruir a sentença preservando
        #    os espaços do texto original
        nova_sentenca = []
        ultima_posicao = 0

        for ocorrencia, (original, previsto) in zip(
            ocorrencias,
            previsoes
        ):

            # adiciona tudo que estava entre o token
            # anterior e o token atual
            nova_sentenca.append(
                sentenca[ultima_posicao:ocorrencia.start()]
            )

            # se o modelo encontrou uma normalização
            if previsto != "":
                nova_sentenca.append(previsto)

                soma += 1

                # guarda exemplos
                if (
                    len(exemplos) < 10
                    and (original, previsto) not in exemplos
                ):
                    exemplos.append(
                        (original, previsto)
                    )

            else:
                # mantém o token original
                nova_sentenca.append(original)

            # atualiza a posição
            ultima_posicao = ocorrencia.end()

        # adiciona o restante da sentença
        nova_sentenca.append(
            sentenca[ultima_posicao:]
        )

        resultado.append(
            "".join(nova_sentenca)
        )

    # 4. Mostrar resultados
    print("\n" + "Total de tokens normalizados:", soma)
    print("\nExemplos de tokens normalizados:")

    for original, normalizado in exemplos:
        print(f"{original} -> {normalizado}")

    return resultado

# ==================================================
# APLICAÇÃO DO MELHOR MODELO
# ==================================================

print("\n" + "=" * 50)
print("NORMALIZAÇÃO COM O MELHOR MODELO")
print("=" * 50)

'''
print(
    "Modelo:",
    melhor_resultado["Modelo"]
)

print(
    "Partição:",
    melhor_resultado["Partição"]
)

print(
    "F-score:",
    round(melhor_resultado["F-score"], 2)
)
'''
# normalização da transcrição
transcricao_norm = normalizar(
    transcricao_pre,
    melhor_modelo
)

# normalização dos tweets
tweets_norm = normalizar(
    tweets_pre,
    melhor_modelo
)

# ==================================================
# VISUALIZAÇÃO DOS TEXTOS NORMALIZADOS
# ==================================================

print("\n" + "=" * 50)
print("TRANSCRIÇÃO")
print("=" * 50)
print("\nExemplos de sentenças normalizadas:")

for i in range(5):
    print("\nORIGINAL:")
    print(transcricao_da[i])

    print("PRÉ-NORMALIZADO:")
    print(transcricao_pre[i])

    print("NORMALIZADO:")
    print(transcricao_norm[i])

print("\n" + "=" * 50)
print("TWEETS")
print("=" * 50)
print("\nExemplos de sentenças normalizadas:")

for i in range(5):
    print("\nORIGINAL:")
    print(raw_tweets[i])

    print("PRÉ-NORMALIZADO:")
    print(tweets_pre[i])

    print("NORMALIZADO:")
    print(tweets_norm[i])