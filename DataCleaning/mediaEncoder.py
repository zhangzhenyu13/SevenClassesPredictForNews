from DataCleaning.ClusteringModel import MediaCluster
from LanguageModel.LDAModel import LDAFlow
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle

class MediaEncoder(object):
    def __init__(self,build=False):
        self.clusterModel=MediaCluster()
        self.clusterModel.loadModel()

        self.languageModel=LDAFlow()
        self.languageModel.name="media"
        self.languageModel.loadModel()

        self.build=build
        if self.build:
            self.onehotModel=OneHotEncoder()
        else:
            with open("../models/one-hot-media.pkl","rb") as f:
                self.onehotModel=pickle.load(f)
            print("load one-hot encoder")

    def encodeMedia(self,media_desc):

        #print("one-hot encoding {} items".format(len(media_desc)))

        lda_x=self.languageModel.transformVec(media_desc)
        clusters=self.clusterModel.model.predict(lda_x)

        clusters=np.reshape(clusters,newshape=(len(clusters),1))

        if self.build:
            self.onehotModel.fit(clusters)
            encoded=self.onehotModel.transform(clusters).toarray()

            with open("../models/one-hot-media.pkl","wb") as f:
                pickle.dump(self.onehotModel,f)
            print("building mode finished")
        else:
            encoded=self.onehotModel.transform(clusters).toarray()

        return encoded


if __name__ == '__main__':
    from DataCleaning.DataLoader import NewsDataset
    nwdata = NewsDataset("../data/trainex.xls")

    nwdata.gatherMedia()

    media_desc=list(nwdata.media.values())

    media_encoder=MediaEncoder(True)
    #print(10*"*"+"\n",media_desc,"\n")
    encode_result=media_encoder.encodeMedia(media_desc)
    print(encode_result.shape)