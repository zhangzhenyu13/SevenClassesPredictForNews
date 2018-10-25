from gensim import corpora
import gensim
import time
import _pickle as pickle
from scipy import sparse
from LanguageModel import WordTokenizer

class LDAFlow:
    def __init__(self):
        self.n_features=100
        self.name=""
        self.cleaner=WordTokenizer.initTokenizer()

    def transformVec(self,docs_list):

        docs=self.cleaner.cleanDocs(docs_list)

        #print("transform {} docs to latent space of lda vectors({})".format(len(docs_list),self.n_features))

        X = sparse.dok_matrix((len(docs), self.n_features))
        row = 0

        for doc in docs:
            doc_bow = self.dictionary.doc2bow(doc)
            lda_doc = self.lda[doc_bow]
            for topic in lda_doc:
                X[row, topic[0]] = topic[1]
            row += 1
        return X.toarray()

    def train_doctopics(self,docs):
        #print(np.shape(docs),docs[0])
        t0 = time.time()
        #print("transfering docs to LDA topics distribution")
        docs = self.cleaner.cleanDocs(docs)
        #self.n_features=min(int(0.9*len(docs[0])),self.n_features)
        print("performing LDA(%d features) from shape(%d,%d)"%(self.n_features,len(docs),len(docs[0])))
        self.dictionary = corpora.Dictionary(docs)
        doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in docs]
        self.lda = gensim.models.LdaModel(doc_term_matrix, num_topics=self.n_features, id2word=self.dictionary)
        print("LDA built in %fs" % (time.time() - t0))
        with open("../models/"+self.name+"-ldamodel.pkl","wb") as f:
            model={}
            model["n_features"]=self.n_features
            model["dict"]=self.dictionary
            model["lda"]=self.lda
            pickle.dump(model,f,True)
        print()


    def loadModel(self):
        print("loading lda model")
        with open("../models/"+self.name+"-ldamodel.pkl","rb") as f:
            model=pickle.load(f)
            self.n_features=model["n_features"]
            self.dictionary=model["dict"]
            self.lda=model["lda"]
            print("loaded %d feature model"%self.n_features)
        print()

if __name__ == '__main__':
    from DataCleaning import DataLoader

    nwdata= DataLoader.NewsDataset("../data/trainex.xls")
    nwdata.gatherMedia()
    ldamodel=LDAFlow()
    ldamodel.name="media"

    mediaName,mediaDesc=[],[]
    for k,v in nwdata.media.items():
        mediaDesc.append(v)
        mediaName.append(k)
    print(mediaDesc)
    ldamodel.train_doctopics(mediaDesc)
