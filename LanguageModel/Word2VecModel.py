# coding=utf-8
import numpy as np
import gensim
import time
from LanguageModel.WordTokenizer import initTokenizer
from DataCleaning.DataLoader import NewsDataset
import collections


class WordEmbedding:
    def __init__(self,embedding_dim,max_words):
        self.feature_num=embedding_dim
        self.clip_length=max_words
        self.tokenizer=initTokenizer()
        self.model = gensim.models.Word2Vec(size=self.feature_num,workers=16, window=10,min_count=5)


        print("init word model")


    def trainDocModel(self, docs,epoch_num=50):
        t0=time.time()

        corpo_docs=self.tokenizer.cleanDocs(docs)

        self.model.build_vocab(corpo_docs)


        self.model.train(corpo_docs,total_examples=len(docs),epochs=epoch_num)

        t1=time.time()
        print("word2vec model training finished in %d s"%(t1-t0))

    def transformDoc2Vec(self,docs):
        print("generate word embeddings")
        embeddings=[]

        corporus_docs=self.tokenizer.cleanDocs(docs)

        for corporus_doc in corporus_docs:
            embedding=np.zeros(shape=(self.clip_length,self.feature_num))
            n_count=min(self.clip_length,len(corporus_doc))
            for i in range(n_count):
                word=corporus_doc[i]
                if word in self.tokenizer.stopwords:
                    continue

                try:
                    wordvec=self.model[word]
                except:
                    continue

                embedding[i]=wordvec


            embeddings.append(embedding)

        embeddings=np.array(embeddings)

        return embeddings


    def saveModel(self):
        self.model.save("models/word2vec")
        print("saved word2vec model")

    def loadModel(self):
        self.buildVoca=False
        self.model=gensim.models.Doc2Vec.load("models/word2vec")
        print("loaded word2vec model")

def trainWordModel():

    data = NewsDataset("data/trainex.xls")



    titles,_=data.fetchAll(cols=[1])
    texts,_=data.fetchAll(cols=[3])

    docs=[str(s[0]) for s in titles]

    for txt in texts:
        doc=txt[0]
        ss=doc.split("。")
        docs+=[str(s) for s in ss]
    types=[type(s) for s in docs]

    print(collections.Counter(types))
    docs=np.array(docs,dtype=str)
    print("all texts itme {}".format(len(docs)))


    wordModel = WordEmbedding(256,100)
    wordModel.trainDocModel(docs,100)
    print(wordModel.model.wv.vocab)
    #exit(10)
    wordModel.saveModel()
    wordModel.loadModel()
    vecs=wordModel.transformDoc2Vec(docs[:10])
    print(vecs.shape)


if __name__ == '__main__':
    trainWordModel()

