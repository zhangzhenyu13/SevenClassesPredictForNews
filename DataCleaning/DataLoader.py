import xlrd
from DataCleaning import crawPics
import pickle
import numpy as np
import pandas as pd
import collections

class NewsDataset(object):

    def readData(self,filename):
        '''
        load an xls file
        :param filename:
        :return:
        '''

        xls_file = xlrd.open_workbook(filename)
        sheetnames = xls_file.sheet_names()
        data = []

        for sn in sheetnames:
            data.append(pd.read_excel(filename,sheet_name=sn).fillna(" "))

        data=pd.concat(data)
        #print(data.head(5))
        data=pd.DataFrame(data)
        #print(data.head(5))
        self.data=data.reset_index(drop=True)

        labels=collections.Counter(self.data.iloc[:,0])

        #print(self.data.head(5))
        #print(self.data.iloc[:,0])
        print(labels)
        self.label2num={}
        self.num2label={}

        index=0
        for k in labels:
            self.label2num[k]=index
            self.num2label[index]=k

            index+=1

        print("loaded {} samples:\n".format(len(self.data)))


    def initPointer(self):
        self.__p=0
        self.data=self.data.sample(frac=1)

    def __init__(self,filename):

        self.readData(filename)
        self.initPointer()
        #print("test loader\n",self.data.iloc[:1000,0])

    def fetchBatch(self,batch_size=20,cols=[1,2,3,4,5,6]):

        left=self.__p
        right=left+batch_size
        n_total=len(self.data)
        diff=right-n_total

        if diff>0:
            rows = [i for i in range(left,n_total)]
            rows+=[i for i in range(diff)]
            self.__p=diff
        else:
            rows=[i for i in range(left,right)]
            self.__p=right

        batch=pd.DataFrame(self.data.iloc[rows,cols].astype("str"))
        labels=[self.label2num[l] for l in self.data.iloc[rows,0]]

        #batch=np.array(batch,dtype=str)
        labels=np.array(labels,dtype=np.int)

        #print("fetch a batch of size{},{}".format(batch.shape,labels.shape))
        return batch,labels


    def fetchAll(self,cols=[1,2,3,4,5,6]):
        X = pd.DataFrame(self.data.iloc[:,cols].astype("str"))
        labels = [self.label2num[l] for l in self.data.iloc[:,0]]

       # X = np.array(X, dtype=str)
        labels = np.array(labels, dtype=np.int)

        print("fetch all train: size{},{}".format(X.shape,labels.shape))
        return X, labels

    def gatherTags(self):
        self.tags=set()

        for k in self.data.keys():
            sheet=self.data[k]
            tags=pd.DataFrame(sheet.iloc[:,4].dropna().astype(str))
            tags=np.reshape(np.array(tags),newshape=len(tags)).tolist()
            #print(tags)
            tags=set(tags)
            self.tags=self.tags|tags
            print("tags size",len(self.tags))

        #print(self.tags)
        self.tags="，".join(self.tags).split("，")
        #self.tags.remove("")
        #print(self.tags)

        with open("../data/newwords","wb") as f:
            pickle.dump(self.tags,f)

    def gatherMedia(self):
        self.media={}
        mediaCount={}
        media={}

        for k in self.data.keys():
            sheet=self.data[k]
            medianame=pd.DataFrame(sheet.iloc[:,5].astype(str))
            mediadisc=pd.DataFrame(sheet.iloc[:,6].astype(str))
            n = len(medianame)
            medianame=np.reshape(np.array(medianame),newshape=n)
            mediadisc=np.reshape(np.array(mediadisc),newshape=n)

            #print(medianame)
            #print(mediadisc)
            #print(n)

            for i in range(n):

                mn=medianame[i]
                if mn not in media.keys():
                    media[mn]=mediadisc[i]
                    mediaCount[mn]=1
                else:
                    mediaCount[mn]=1+mediaCount[mn]

        #print(len(mediaCount),mediaCount)
        values=list(mediaCount.values())
        values.sort(reverse=True)
        #print(values)

        for k in mediaCount.keys():
            if mediaCount[k]<3:
                continue

            self.media[k]=media[k]

        print("media num",len(self.media))

    def compareMedia(self):
        media={}
        for k in self.data.keys():
            media[k]=set(self.data[k].iloc[:,5])
        for k1 in media.keys():
            for k2 in media.keys():
                if k1==k2:
                    continue
                m1,m2=media[k1],media[k2]

                print("common between {}({}) and {}({}) is {}".
                      format(k1,len(m1),k2,len(m2),len(m1.intersection(m2))))

    def gatherPictures(self):
        for k in self.data.keys():
            urls=self.data[k].iloc[:,2]
            crawPics.PAGE_URL_LIST=urls
            crawPics.main()

if __name__ == '__main__':

    nwdata=NewsDataset("../data/trainex.xls")
    #nwdata.gatherTags()
    #nwdata.compareMedia()
    #nwdata.gatherMedia()
    X,Y=nwdata.fetchAll([1,3])

    [print(len(s),s) for s in X.iloc[:10,0]]
    [print(len(s),s) for s in X.iloc[:10,1]]

    X=list(nwdata.data.iloc[:,[1,3]])
    [print(len(s), s) for s in X[:10, 0]]
    [print(len(s), s) for s in X[:10, 1]]