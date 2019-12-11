from Data import Data
import matplotlib.pyplot
import time

SSE = [] # sum of the squared errors
TestUpperBound = 15
for k in range(1,TestUpperBound):
    # print('Now @ k = {0}'.format(k))
    start=time.time()
    testK = Data()
    testK.ReadData('data.csv')
    result = testK.KMeans(k)
    result.ShowLabelInfo(output=False)
    time1=time.time()-start
    silScore = result.getScore(method='Silhouette')
    calScore = result.getScore(method='CalinskiHarabasz')
    davScore = result.getScore(method='DaviesBouldin')
    
    print('TIME:{:.2f}'.format(time1),'K:',k,'SIL:{:.4f}'.format(silScore),'CAL:{:.4f}'.format(calScore),'DAV:{:.4f}'.format(davScore),'CNT:',sum(result.distributionInfo['Num']),'LC',result.distributionInfo['Num'])
    SSE.append(testK.midResult.inertia_)

x = range(1,TestUpperBound)
matplotlib.pyplot.figure(figsize=(5,5))
matplotlib.pyplot.xlabel('k')
matplotlib.pyplot.ylabel('SSE')  
matplotlib.pyplot.plot(x,SSE,'o-')  
matplotlib.pyplot.show()

