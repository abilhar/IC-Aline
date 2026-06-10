# IC Aline
Estudo de iniciação científica na área de PLN 
Desenvolvimento de algoritmos de normalização de textos com marcas de oralidade: tweets (Corpus Carolina USP) e transcrição de conversa adulto-criança (Corpus CEDAE Unicamp).
O processo de normalização tem como objetivo substituir palavras out-of-vocabulary (OOV) -- fora do léxico padrão da língua -- por palavras padronizadas.
O primeiro algoritmo desenvolvido foi baseado em substituição simples de palavras OOV por sua normalização a partir de um dicionário de substituições criado manualmente. 
O segundo algoritmo foi desenvolvido usando unigram e bigram taggers que foram treinados com as normalizações como "etiquetas" das palavras OOV, enquanto as palavras que não deveriam ser normalizadas receberam uma etiqueta vazia. 
