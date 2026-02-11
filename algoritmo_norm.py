import re
import csv
import nltk

#carregamento de corpora e dicionário
#corpora = raw_tweets and transc_da
#lexico pt-br para base
with open("dadosteste100.csv", "r", encoding="utf-8") as tweets:
    raw_tweets = tweets.read().splitlines()

with open("transc_da.txt", "r", encoding="utf-8") as transc:
    transcricao_da = transc.read().splitlines()

with open("lexporbr.csv", "r", encoding="utf-8") as lex:
    lexico = set(lex.read().splitlines())

#dicionário de substituições
with open("substituicoes.csv", "r", encoding="utf-8") as f:
    next(f)
    reader = csv.reader(f, skipinitialspace=True)
    dict_subs = dict(reader)


#limpeza de links e lower case

transcricao_da = [w.lower() for w in transcricao_da]

raw_tweets = [re.sub(r"https?:\/\/\S+", "", item) for item in raw_tweets]
raw_tweets = [w.lower() for w in raw_tweets]

#meu_dict.update({'direitu': "direito", "quardá": "guardar", "ôvinho", "ovinho"})


#Regex para manter emails, hashtags e nomes de usuário sem alteração
#e separar pontuação de palavras
TOKEN_REGEX = (
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"  # email
    r"|@[A-Za-z0-9_]+"                                 # username (inclui @_...)
    r"|#[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+"                        # hashtag
    r"|[A-Za-zÀ-ÖØ-öø-ÿ]+"                             # palavra
    r"|."                                              # qualquer outro caractere
)

#excluir letras repetidas quando necessário (obrigadooo, mtooo)
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


tweets_limpos = [excluir_letras_repetidas(t) for t in raw_tweets]

#print(tweets_limpos[1424])

todos_itens = transcricao_da + tweets_limpos

#função de normalização baseado em substituição

def normalizar(texto):
    partes = re.findall(TOKEN_REGEX, texto)
    partes = [
        p if p not in dict_subs else dict_subs[p]
        for p in partes
    ]
    return "".join(partes)

tweets_norm = [normalizar(t) for t in tweets_limpos]
transcricao_norm = [normalizar(s) for s in transcricao_da]

#print(transcricao_norm[:65])
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
