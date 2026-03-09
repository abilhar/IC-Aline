import re
import csv
import nltk

#carregamento de dados anotados
with open("dados_controle.csv", "r", encoding="utf-8") as tweets:
    tweets_controle = tweets.read().splitlines()

with open("transc_da_anotada.txt", "r", encoding="utf-8") as transcricao:
    transcricao_controle = transcricao.read().splitlines()

tweets_controle = tweets_controle[1:]
transcricao_controle = transcricao_controle[16:]

#carregamento de corpora e dicionário
#corpora = raw_tweets and transc_da
#lexico pt-br para base
with open("dadosteste2000.csv", "r", encoding="utf-8") as tweets:
    raw_tweets = tweets.read().splitlines()

with open("transc_da.txt", "r", encoding="utf-8") as transc:
    transcricao_da = transc.read().splitlines()

transcricao_da = transcricao_da[14:]
raw_tweets = raw_tweets[1:]

with open("lexporbr.csv", "r", encoding="utf-8") as lex:
    lexico = set(lex.read().splitlines())

#retirar links e transformar tudo em lower case
tweets_controle = [re.sub(r"https?:\/\/\S+", "", item) for item in tweets_controle]
tweets_controle = [line.lower() for line in tweets_controle]

transcricao_controle = [line.lower() for line in transcricao_controle]

#Regex para manter emails, hashtags e nomes de usuário sem alteração
#e separar pontuação de palavras mantendo as identificações de pessoas na transcrição
TOKEN_REGEX = (
    r"[:;=][()DdPp]+"                                    #emoticons comuns que não devem ser separados
    r"|[A-Za-zÀ-ÖØ-öø-ÿ_]+/[A-Za-zÀ-ÖØ-öø-ÿ_]+"          #anotações
    r"|p\.:"
    r"|d\.:"
    r"|a\.:"                                              #letras de identificação
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"    #emails
    r"|@[A-Za-z0-9_]+"                                    #nomes de usuário 
    r"|#[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+"                           #hashtags
    r"|[A-Za-zÀ-ÖØ-öø-ÿ_]+"                                 #outras palavras
    r"|."                                                 #outras pontuações
)

#excluir letras repetidas quando necessário para incluir abreviações com letras repetidas desnecessárias
# e normalizar palavras do léxico que foram escritas de maneira alongada (obrigadooo, mtooo)
def excluir_letras_repetidas(texto):
    partes = re.findall(TOKEN_REGEX, texto)
    novas = []

    for p in partes:
        if p.startswith("@") or p.startswith("#") or "@" in p and "." in p:
            novas.append(p)
        elif p in lexico:
            novas.append(p)
        elif p.isalpha():
            novas.append(
                re.sub(r"([A-Za-zÀ-ÖØ-öø-ÿ])\1+", r"\1", p)
            )
        else:
            novas.append(p)

    return "".join(novas)

#criação de dados standard
#tirar letras repetidas

tweets_controle = [excluir_letras_repetidas(t) for t in tweets_controle]
transcricao_controle = [excluir_letras_repetidas(t) for t in transcricao_controle]

def criar_tuplas(texto):
    tokens = [t for t in re.findall(TOKEN_REGEX, texto) if not t.isspace()]
    
    resultado = []

    for token in tokens:
        if "/" in token:
            original, substituicao = token.split("/")
            resultado.append((original, substituicao))
        elif token.isalpha():
            resultado.append((token, ""))
        else:
            resultado.append((token, ""))

    return resultado

dados_base = []

for item in tweets_controle:
    dados_base.extend(criar_tuplas(item))

for item in transcricao_controle:
    dados_base.extend(criar_tuplas(item))




#dicionário de substituições
with open("substituicoes.csv", "r", encoding="utf-8") as f:
    next(f)
    reader = csv.reader(f, skipinitialspace=True)
    dict_subs = dict(reader)


#limpeza de links e lower case

transcricao_da = [line.lower() for line in transcricao_da]

raw_tweets = [re.sub(r"https?:\/\/\S+", "", item) for item in raw_tweets]
raw_tweets = [line.lower() for line in raw_tweets]

#alterações posteriores ao dicionário
dict_subs.update({'direitu': "direito", "quardá": "guardar", "ôvinho": "ovinho", "to": "estou", "ta": "está"})
del dict_subs["sp"]

#função de normalização baseado em substituição

def normalizar(texto):
    partes = re.findall(TOKEN_REGEX, texto)
    partes = [
        p if p not in dict_subs else dict_subs[p]
        for p in partes
    ]
    return "".join(partes)

#normalização antes de tirar letras repetidas para incluir casos de abreviações como "vdd" e "add"
tweets_norm1 = [normalizar(t) for t in raw_tweets]
transcricao_norm = [normalizar(s) for s in transcricao_da]

#print(transcricao_norm[:50])



#print(raw_tweets[35])


tweets_limpos = [excluir_letras_repetidas(t) for t in tweets_norm1]

#print(tweets_limpos[35])

#segunda normalização dos tweets, agora sem as letras repetidas desnecessárias
tweets_norm = [normalizar(t) for t in tweets_limpos]

#print(tweets_norm[925])
#conferindo se há mais palavras fora do lexico 
'''
palavras = []
for string in transcricao_da:
    partes = re.findall(TOKEN_REGEX, string)
    for w in partes:
        if w.isalpha():
            if w not in lexico:
                if w not in dict_subs:
                    palavras.append(w)
                    
for string in tweets_limpos:
    partes = re.findall(TOKEN_REGEX, string)
    for w in partes:
        if w.isalpha():
            if w not in lexico:
                if w not in dict_subs:
                    palavras.append(w)

print(set(palavras))
'''

#Cálculos estatísticos
#definições de VP, VN, FP, FN
#positivo = normalização correta
#VP/TP => A palavra deveria ser normalizada e o sistema normalizou corretamente
#VN/TN => A palavra não deveria ser normalizada e o sistema não normalizou
#FP => A palavra não deveria ser normalizada e o sistema normalizou
#FN => A palavra deveria ser normalizada e o sistema errou a normalização ou não normalizou

#os dados_base são uma lista das palavras dos corpora e suas respectivas substituições. Quando nenhuma normalização deveria ser feita, o segundo item da tupla é uma string vazia

#separar resultados da normalização em tokens para comparação
tokens_previstos = []

for frase in tweets_norm + transcricao_norm:
    tokens_previstos.extend(re.findall(TOKEN_REGEX, frase))
    
tokens_previstos = [t for t in tokens_previstos if t.strip() != ""]

print(len(dados_base))
print(len(tokens_previstos))

#cálculos de VP, VN, FP, FN
def calcular_metricas(dados, predicoes):

    tp = tn = fp = fn = 0

    for (original, esperado), previsto in zip(dados, predicoes):

        if esperado != "" and previsto == esperado:
            tp += 1

        elif esperado == "" and previsto == original:
            tn += 1

        elif esperado == "" and previsto != original:
            fp += 1

        elif esperado != "" and previsto != esperado:
            fn += 1

    return tp, tn, fp, fn

tp, tn, fp, fn = calcular_metricas(dados_base, tokens_previstos)

print(tp, tn, fp, fn)

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print("Accuracy:", accuracy*100)
print("Precision:", precision*100)
print("Recall:", recall*100)
print("F1:", f1*100)

'''
#conferindo dados não sincronizados
for i, ((orig, esp), prev) in enumerate(zip(dados_base, tokens_previstos)):

    esperado_final = esp if esp != "" else orig

    if esperado_final != prev:
        print(i, orig, esp, prev)
        input("Press Enter to see the next item...")
'''

