from sklearn.cluster import KMeans
import pickle
import numpy as np


class MediaCluster(object):
    def __init__(self):
        self.name="media"
        self.cluster_num=50
        self.centers=None
        self.model=None

    def buildModel(self,X):

        data={"model":None,"centers":None}
        self.model=KMeans(n_clusters=self.cluster_num)
        self.model.fit(X)
        self.centers=self.model.cluster_centers_
        with open("../models/"+self.name+"-cluster.pkl","wb") as f:
            data["model"]=self.model
            data["centers"]=self.centers
            pickle.dump(data,f)

    def predictDistVec(self,X):
        X=np.array(X)
        dist=lambda x: [np.sum(np.square(x-p)) for p in self.centers]
        dist_vecs=[]
        for x in X:
            dist_vecs.append(dist(x))

        return dist_vecs

    def loadModel(self):
        with open("../models/"+self.name+"-cluster.pkl","rb") as f:
            data=pickle.load(f)
            self.model=data["model"]
            self.centers=data["centers"]

def generateMediaClusters():
    clusterModel=MediaCluster()
    clusterModel.buildModel(lda_media_desc)
    clusters=clusterModel.model.predict(lda_media_desc)

    media_data={}
    clusterCount={i:0 for i in range(clusterModel.cluster_num)}
    for i in range(len(clusters)):
        name=mediaName[i]
        media_data[name]=clusters[i]
        clusterCount[clusters[i]]+=1

    with open("../data/mediaCls.pkl","wb") as f:
        pickle.dump(media_data,f)
    print(media_data)
    print(clusterCount)

    dists=clusterModel.predictDistVec(lda_media_desc[:5])
    print("tuning")
    print(dists[:5])
    print(clusters[:5])
    print(np.argmax(dists,axis=1)[:5])

if __name__ == '__main__':
    from DataCleaning import DataLoader
    from LanguageModel.LDAModel import LDAFlow
    nwdata = DataLoader.NewsDataset("../data/trainex.xls")
    nwdata.gatherMedia()
    ldamodel = LDAFlow()
    ldamodel.name = "media"
    ldamodel.loadModel()

    mediaName, mediaDesc = [], []

    for k, v in nwdata.media.items():
        mediaDesc.append(v)
        mediaName.append(k)

    lda_media_desc=ldamodel.transformVec(mediaDesc)
    generateMediaClusters()
