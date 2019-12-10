from Data import Data
import matplotlib.pyplot, numpy, matplotlib.pyplot

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score,calinski_harabasz_score

testK = Data()
testK.ReadData('data.csv')


def getEsp(K_Neighbors,DiscardLast=0):
    neigh = NearestNeighbors(n_neighbors=K_Neighbors+1)
    nbrs = neigh.fit(numpy.array(testK.data))
    distances,_ = nbrs.kneighbors(testK.data)
    distances = numpy.sort(distances, axis=0)
    distances = distances[:,K_Neighbors]
    if type(DiscardLast) is int:
        distances = distances[:-DiscardLast]
    elif type(DiscardLast) is float:
        distances = distances[:-int(DiscardLast*len(distances))]
    callDerivative = lambda value: list(map(lambda x: x[1]-x[0], zip(value,value[0] + value[:-1])))
    distancesDD = callDerivative(callDerivative(distances))
    mx = max(distancesDD)
    for i in range(len(distancesDD)-1, -1, -1):
        if distancesDD[i]==mx:
            return i,distances[i]
    return None

def getSilhouetteScore(data,label):
    if max(label) == 0:
        return -2
    removedData = list(filter(lambda x:x[1]>=0, zip(data,label)))
    return silhouette_score(list(map(lambda x:x[0],removedData)),list(map(lambda x:x[1],removedData)))

def getCalinskiHarabaszScore(data,label):
    if max(label) == 0:
        return -2
    removedData = list(filter(lambda x:x[1]>=0, zip(data,label)))
    return calinski_harabasz_score(list(map(lambda x:x[0],removedData)),list(map(lambda x:x[1],removedData)))


for i in range(90,280,10):
# for i in range(145,280,5):
# for i in range(155,165,1):# 160 makes silhouette score max
    kn = i
    esp = getEsp(kn,DiscardLast=0.1)[1]
    test = Data()
    test.ReadData('data.csv')
    result = test.DBSCAN(eps=esp,min_samples=kn)
    result.ShowLabelInfo(output=False)
    silScore = getSilhouetteScore(result.data,result.predict)
    calScore = getCalinskiHarabaszScore(result.data,result.predict)
    print('K:',kn,'ESP:{:.4f}'.format(esp),'SIL:{:.4f}'.format(silScore),'CAL:{:.4f}'.format(calScore),'CNT:',sum(result.distributionInfo['Num']),'LB:',len(result.distributionInfo['Num']),'LC',result.distributionInfo['Num'])
