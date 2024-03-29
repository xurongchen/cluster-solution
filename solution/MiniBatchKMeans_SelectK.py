from Data import Data
import matplotlib.pyplot

TestUpperBound = 15
test = Data()
test.ReadData('data.csv')
test = test.pca(0.95)
SSE = [] # sum of the squared errors
for k in range(1,TestUpperBound):
    testK = test.Copy()
    result = testK.MiniBatchKMeans(k)
    result.ShowLabelInfo(output=False)
    silScore = result.getScore(method='Silhouette')
    calScore = result.getScore(method='CalinskiHarabasz')
    davScore = result.getScore(method='DaviesBouldin')
    print('K:',k,'SIL:{:.4f}'.format(silScore),'CAL:{:.4f}'.format(calScore),'DAV:{:.4f}'.format(davScore),'CNT:',sum(result.distributionInfo['Num']),'LC',result.distributionInfo['Num'])
    SSE.append(testK.midResult.inertia_)

x = range(1,TestUpperBound)
matplotlib.pyplot.figure(figsize=(5,5))
matplotlib.pyplot.xlabel('k')
matplotlib.pyplot.ylabel('SSE')  
matplotlib.pyplot.plot(x,SSE,'o-')  
matplotlib.pyplot.show()
