import torch
from Classifiers.StackingRNN import StackingRNN
from Classifiers.AttEncoder import AttnEncoderNet,EncoderRNN,Attn
from Classifiers.CNNText import TextCNN
from Classifiers.Trainer import ModelTrainer
from DataCleaning.DataLoader import NewsDataset
import collections
import numpy as np
from LanguageModel.VocManager import *
import copy

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

#config
MAX_LENGTH = [10,20]

model_name = 'attn_encoder_net'

attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
epoch_num=10000
embedding_size=64
hidden_size = 64
n_layers = 2
dropout = 0.5
batch_size = 64
num_classes=7

train_cols=[1,3]
test_cols=[3,7]
# Set checkpoint to load from; set to None if starting from scratch
loadModel,loadVoc=False,True
trainData, testData, voc, tokenizer_ch = None,None,None,None
# prepare and run model funcs

def testAcc(model:StackingRNN,testData:NewsDataset):

    #testX1, testY = testData.fetchBatch(batch_size,test_cols)
    n_data=len(testData.data)
    print("test {} samples".format(n_data))
    count=n_data//batch_size

    #model.evalMode()

    results=[]
    trues=[]
    for i in range(count+1):
        testX2,testY=testData.fetchBatch(batch_size,test_cols)
        trues.append(testY)

        testX1 = testX2.iloc[:,0]#["。".join(one) for one in testX1]
        testX2=testX2.iloc[:,1]
        testX1 = genSeqs(testX1, voc, tokenizer_ch,0)
        testX2 = genSeqs(testX2, voc, tokenizer_ch,1)

        testX1 = torch.from_numpy(testX1)
        testX2 = torch.from_numpy(testX2)
        predicts=model(testX1,testX2)
        predicts=torch.argmax(predicts,dim=1).numpy()


        #print("test batch",np.sum(equals))
        results.append(predicts)

    results=np.concatenate(results,axis=0)[:n_data]
    trues=np.concatenate(trues,axis=0)[:n_data]
    print("for test(predicts and true distribution)",collections.Counter(results),collections.Counter(trues))
    #print(results.shape,trues.shape,"\n",results[:10],"\n",trues[:10])
    equals=np.sum(results==trues)

    print("\ncorrectly predicted {}:{}/{}\n".format(float(equals)/n_data,equals,n_data))

    return float(equals)/n_data


def trainRun(trainer:ModelTrainer,model:StackingRNN,trainData:NewsDataset):
    bestModel=None
    bestAcc=0.0
    n_data=len(trainData.data)
    print("train with {} samples for {} epochs".format(n_data,epoch_num))

    count=1
    for iter_num in range(epoch_num):

        batch,labels=trainData.fetchBatch(batch_size=batch_size,cols=train_cols)
        docs1=batch.iloc[:,0]#["。".join(one) for one in batch]
        docs2=batch.iloc[:,1]
        X1=genSeqs(docs1,voc,tokenizer_ch,0)
        X2=genSeqs(docs2,voc,tokenizer_ch,1)

        model.trainMode()

        trainer.zeroGrad()
        loss=trainer.getLoss((X1,X2),labels)
        loss.backward()
        trainer.optimizeModel()

        #print("#{}:loss=".format(iter_num+1),loss)


        if (iter_num+1)%20==0:
            model.evalMode()
            acc=testAcc(model,testData)
            if acc>bestAcc:
                bestModel=copy.deepcopy(model)
                bestAcc=acc
            print("best acc of the model", bestAcc)

        if batch_size*(iter_num+1)/n_data>=count:
            print("shuffle training data")
            trainData.initPointer()
            count+=1

    print("\nbest acc of the model",bestAcc)

def genSeqs(docs,voc:Voc,tokenizer:TokenizerCh,choice):
    cleaned=tokenizer.cleanDocs(docs)

    seqs=[]

    for tks in cleaned:
        seq=np.zeros(shape=MAX_LENGTH[choice],dtype=np.long)
        count=0
        for w in tks:
            if count>=MAX_LENGTH[choice]:
                break
            if w in voc.word2count.keys():
                seq[count]=voc.word2index[w]
            else:
                pass
                #seq[count]=UNK_token

            count+=1

        seqs.append(seq)

    seqs=np.array(seqs,dtype=np.long)

    return seqs





print("init "+model_name,"batch size",batch_size)
#load train data
def loadData():
    trainData=NewsDataset("data/trainex.xls")
    testData=NewsDataset("data/testex.xls")
    testData.label2num,testData.num2label=trainData.label2num,trainData.num2label
    print(testData.label2num)
    print(trainData.label2num)
    tokenizer_ch=initTokenizer()

    if loadVoc:
        voc=Voc("news_voc")
        voc.loadVoc()
    else:
        voc=buildVoc(tokenizer_ch,trainData)
        voc.trim(10)
    return trainData,testData,voc,tokenizer_ch


def runModel():


    print('Building encoder and decoder ...')

    #encoder = EncoderRNN(hidden_size, voc.num_words, n_layers, dropout)
    #model = AttnEncoderNet(encoder,attn_model, hidden_size, num_classes, n_layers, dropout)

    #model=StackingRNN(embedding_size,hidden_size,num_classes, voc.num_words, n_layers, dropout)
    model=TextCNN(max_len=MAX_LENGTH, embedding_size=embedding_size, classes=num_classes, num_words=voc.num_words)
    if loadModel:
        model.loadModel()

    trainer = ModelTrainer(model)

    # Use appropriate device
    #encoder = encoder.to(device)
    #model = model.to(device)
    print('Models built and ready to go!')


    # Run training iterations
    print("Starting Training!")
    model.trainMode()
    trainRun(trainer,model,trainData)
