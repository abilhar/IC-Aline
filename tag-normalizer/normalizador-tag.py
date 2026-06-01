import re
import csv
import nltk
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import BrillTagger
import numpy as np
import joblib 
from nltk.corpus import mac_morpho
#nltk.download('punkt')

#carregamento de dados anotados
with open("2000tweets-2705.csv", "r", encoding="utf-8") as tweets:
    tweets_controle = tweets.read().splitlines()

with open("transc_da_anotada-2705.txt", "r", encoding="utf-8") as transcricao:
    transcricao_controle = transcricao.read().splitlines()

tweets_controle = tweets_controle[1:]
transcricao_controle = transcricao_controle[16:]

#carregamento de corpora para processamento
#corpora = raw_tweets and transc_da
#raw_data
with open("tweets-teste-2000.csv", "r", encoding="utf-8") as tweets:
    raw_tweets = tweets.read().splitlines()

with open("transc-da-teste.txt", "r", encoding="utf-8") as transc:
    transcricao_da = transc.read().splitlines()

transcricao_da = transcricao_da[14:]
transcricao_da_tokens = [word_tokenize(t) for t in transcricao_da]
raw_tweets = raw_tweets[1:]
raw_tweets_tokens = [word_tokenize(t) for t in raw_tweets]
raw_data = transcricao_da_tokens + raw_tweets_tokens
raw_data = [x for x in raw_data if x]
#print(raw_data[:50])

TOKEN_REGEX = (
    r"[:;=][()DdPp]+"                                    #emoticons comuns que não devem ser separados
    r"|[A-Za-zÀ-ÖØ-öø-ÿ_]+--/[A-Za-zÀ-ÖØ-öø-ÿ_]+"          #anotações
    r"|p\.:"
    r"|d\.:"
    r"|a\.:"                                              #letras de identificação
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"    #emails
    r"|@[A-Za-z0-9_]+"                                    #nomes de usuário 
    r"|#[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+"                           #hashtags
    r"|[A-Za-zÀ-ÖØ-öø-ÿ_]+"                                 #outras palavras
    r"|."                                                 #outras pontuações
)


#organização de dados para treinamento - criação de tuplas
def criar_tuplas(texto):
    tokens = [t for t in re.findall(TOKEN_REGEX, texto) if not t.isspace()]
    
    resultado = []

    for token in tokens:
        if "--/" in token:
            original, substituicao = token.split("--/", 1)
            resultado.append((original, substituicao))

        else:
            resultado.append((token, ""))

    return resultado

#criação de train_data e test_data (80-20)
tuplas_transc = []
tuplas_tweet = []

for item in transcricao_controle:
    tuplas_transc.append(criar_tuplas(item))
    
for item in tweets_controle:
    tuplas_tweet.append(criar_tuplas(item))

split_transc = int(len(tuplas_transc) * 0.8)
split_tweet = int(len(tuplas_tweet) * 0.8)

train_data = (
    tuplas_transc[:split_transc]
    + tuplas_tweet[:split_tweet]
)
train_data = [x for x in train_data if x] #tira listas vazias

test_data = (
    tuplas_transc[split_transc:]
    + tuplas_tweet[split_tweet:]
)
test_data = [x for x in test_data if x] #tira listas vazias

#print(train_data[:50])
#print(test_data[:50])

#treinamento

default = nltk.DefaultTagger('')

uni_tagger = nltk.UnigramTagger(train_data, backoff=default)
bi_tagger = nltk.BigramTagger(train_data, backoff=default)
#brill_tagger = nltk.BrillTagger(train_data)

#teste
print("Performance")
print("Unigram =", uni_tagger.accuracy(test_data)* 100)
print("Bigram =", bi_tagger.accuracy(test_data)* 100)




