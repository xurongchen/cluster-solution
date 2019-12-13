from Data import Data
import matplotlib.pyplot, numpy, matplotlib.pyplot

from sklearn.neighbors import NearestNeighbors

testK = Data()
testK.ReadData('data.csv')

# Notice: pca will affect the DBSCAN a lot
testK = testK.pca(0.95)

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


for i in range(90,280,10):
# for i in range(145,200,5):
# for i in range(155,165,1):# 160 makes silhouette score max
    kn = i
    esp = getEsp(kn,DiscardLast=0.1)[1]
    test = testK.Copy()
    # test.ReadData('data.csv')
    result = test.DBSCAN(eps=esp,min_samples=kn)
    result.ShowLabelInfo(output=False)
    silScore = result.getScore(method='Silhouette')
    davScore = result.getScore(method='DaviesBouldin')
    calScore = result.getScore(method='CalinskiHarabasz')
    print('K:',kn,'ESP:{:.4f}'.format(esp),'SIL:{:.4f}'.format(silScore),'CAL:{:.4f}'.format(calScore),'DAV:{:.4f}'.format(davScore),'CNT:',sum(result.distributionInfo['Num']),'LB:',len(result.distributionInfo['Num']),'LC',result.distributionInfo['Num'])
