from LanguageModel.Word2VecModel import trainWordModel
from Classifiers.SimpleDNN import testTrain
import collections
import numpy as np
#trainWordModel()


#testTrain()

def runModel():
    from Classifiers import runClassifiers

    runClassifiers.trainData, runClassifiers.testData, runClassifiers.voc, runClassifiers.tokenizer_ch=runClassifiers.loadData()
    runClassifiers.runModel()


#from LanguageModel.VocManager import testFunc
#testFunc()

def testLength():
    from LanguageModel.WordTokenizer import initTokenizer
    tokenizer=initTokenizer()
    from DataCleaning.DataLoader import NewsDataset

    nwdata = NewsDataset("data/testex.xls")
    X,Y=nwdata.fetchAll(cols=[3,7])
    X1=X.iloc[:,0]
    X2=X.iloc[:,1]


    print(X.shape,Y.shape,X1.shape,X2.shape)
    length1=[len(s) for s in X1]
    length2=[len(s) for s in X2]
    [print(len(s),s) for s in X2[:5]]
    print("init length")
    print(np.min(length1), np.max(length1), np.mean(length1))
    print(np.min(length2), np.max(length2), np.mean(length2))

    cleaned1=tokenizer.cleanDocs(X1)
    cleaned2=tokenizer.cleanDocs(X2)

    length1=[len(tks) for tks in cleaned1]
    length2=[len(tks) for tks in cleaned2]

    print("tks length")
    print(np.min(length1),np.max(length1),np.mean(length1))
    print(np.min(length2),np.max(length2),np.mean(length2))


#testLength()


runModel()