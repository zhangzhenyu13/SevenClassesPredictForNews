# _*_coding:utf-8 _*_
import numpy as np
import fastText
from DataCleaning.DataLoader import NewsDataset
from  LanguageModel.WordTokenizer import initTokenizer
from LanguageModel.VocManager import *

buildVocorNot=True
def loadData():
    trainData=NewsDataset("data/trainex.xls")
    testData=NewsDataset("data/testex.xls")
    testData.label2num,testData.num2label=trainData.label2num,trainData.num2label
    print(testData.label2num)
    print(trainData.label2num)
    tokenizer_ch=initTokenizer()


    #voc=buildVoc(tokenizer_ch,trainData)
    voc=Voc("news_voc")
    voc.loadVoc()
    voc.trim(5)
    return trainData,testData,voc,tokenizer_ch


def generateRequiredFile(X,Y,filePath):
    with open(filePath,"w",encoding="utf-8") as f:
        titles=X.iloc[:,0]
        contents=X.iloc[:,1]
        n_total=len(X)
        titles=tokenizer.cleanDocs(titles)
        contents=tokenizer.cleanDocs(contents)
        for i in range(n_total):
            title,content,label=titles[i],contents[i],str(Y[i])

            s=" ".join(title)+" "+" ".join(content)+" \t__label__"+label+"\n"
            #s=s.decode("utf-8").encode("utf-8")
            f.write(s)

trainData,testData,voc,tokenizer=loadData()
trainPath="data/train.txt"
testPath="data/test.txt"

#trainX,trainY=trainData.fetchAll(cols=[1,3])
#generateRequiredFile(trainX,trainY,trainPath)

#testX,testY=testData.fetchAll(cols=[3,7])
#generateRequiredFile(testX,testY,testPath)

model=fastText.train_supervised(trainPath,lr=0.01,epoch=100)
results=model.test(testPath)

print(results)


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


classifier = model
labels_right = []
texts = []
testX,testY=testData.fetchAll(cols=[3,7])
titles=testX.iloc[:,0]
contents=testX.iloc[:,1]
n_total=len(testY)
titles=tokenizer.cleanDocs(titles)
contents=tokenizer.cleanDocs(contents)
for i in range(n_total):
    title,content=titles[i],contents[i]
    s=" ".join(title)+" "+" ".join(content)
    texts.append(s.replace("\n"," "))
    label=testY[i]
    labels_right.append(label)


print(len(texts))
labels_predict = []
for text in texts:
    result=classifier.predict(text)
    labels_predict.append(result[0])
# print labels_predict
print(labels_right)
print(labels_predict)
print()
labels_predict=[l[0].replace("__label__","") for l in labels_predict]

labels_right=np.array(labels_right,dtype=np.int)
labels_predict=np.array(labels_predict,dtype=np.int)
acc=np.sum(labels_predict==labels_right)
print(acc,float(acc)/len(labels_right))