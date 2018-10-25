import jieba
import pickle
import numpy as np


class TokenizerCh(object):
    def __init__(self,newwords,stopwords):
        [jieba.add_word(word) for word in newwords]
        self.stopwords=set()
        [self.stopwords.add(word) for word in stopwords]
        self.stopwords.remove("")
        #print("stopwords({}):{}".format(len(self.stopwords),self.stopwords))
        print("init tokenizer")

    def cleanDoc(self,doc):
        #doc.decode("utf-8").encode("utf-8")
        words=jieba.cut(doc)
        keeped_words=[]
        #print(list(words))
        for word in words:
            if word in self.stopwords:
                continue
            keeped_words.append(word)

        return keeped_words

    def cleanDocs(self,docs):

        #print("clean {} docs".format(len(docs)))

        #cld=np.vectorize(self.cleanDoc)
        cleaned=[]
        #cleaned=cld(docs)
        for doc in docs:
            cleaned.append(self.cleanDoc(doc))
        #[print(len(tks)) for tks in cleaned]
        return cleaned

def initTokenizer():
    with open("data/newwords","rb") as f:
        newwords=pickle.load(f)

    with open("data/stopwords.txt","r",encoding="UTF-8'") as f:
        stopwords=f.read()
        stopwords=stopwords.split("\n")
    tk=TokenizerCh(newwords,stopwords)

    return tk

if __name__ == '__main__':
    initTokenizer()
