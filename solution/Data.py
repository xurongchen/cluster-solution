import os,re
import sklearn.cluster
import sklearn.manifold
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
import numpy
import matplotlib.pyplot
import matplotlib.patheffects
import seaborn

import sklearn.preprocessing

class Data:
    
    @staticmethod
    def getSilhouetteScore(data,label):
        if max(label) == 0:
            return -2
        removedData = list(filter(lambda x:x[1]>=0, zip(data,label)))
        return silhouette_score(list(map(lambda x:x[0],removedData)),list(map(lambda x:x[1],removedData)))

    @staticmethod
    def getDaviesBouldinScore(data,label):
        if max(label) == 0:
            return -2
        removedData = list(filter(lambda x:x[1]>=0, zip(data,label)))
        return davies_bouldin_score(list(map(lambda x:x[0],removedData)),list(map(lambda x:x[1],removedData)))

    @staticmethod
    def getCalinskiHarabaszScore(data,label):
        if max(label) == 0:
            return -2
        removedData = list(filter(lambda x:x[1]>=0, zip(data,label)))
        return calinski_harabasz_score(list(map(lambda x:x[0],removedData)),list(map(lambda x:x[1],removedData)))

    def getScore(self,method='Silhouette'):
        if method=='Silhouette':
            return Data.getSilhouetteScore(self.data,self.predict)
        elif method=='CalinskiHarabasz':
            return Data.getCalinskiHarabaszScore(self.data,self.predict)
        elif method=='DaviesBouldin':
            return Data.getDaviesBouldinScore(self.data,self.predict)
        raise NameError('Method name error!')
    
    data = []
    dimensions = []
    labeled = False
    RunStandardScaler = False
    RunNormalize = False

    midResult = None
    predict = None
    distributionInfo = None
    
    
    def ReadData(self, filePath):
        if not os.path.isfile(filePath):
            raise FileNotFoundError('Data file not found!')
        fopen = open(filePath)
        isTitle = True
        self.data = []
        for line in fopen.readlines():
            if isTitle:
                self.dimensions = re.findall(r'"(.*?)"',line)
                isTitle = False
            else:
                self.data.append(tuple(map(int,line.strip().split(','))))
        fopen.close()
        if self.RunStandardScaler:
            self.data = sklearn.preprocessing.StandardScaler().fit_transform(self.data) 
        if self.RunNormalize:
            self.data = sklearn.preprocessing.normalize(self.data)

    def Copy(self):
        result = Data()
        result.data = self.data
        result.labeled = self.labeled
        result.dimensions = self.dimensions
        result.RunNormalize = self.RunNormalize
        result.RunStandardScaler = self.RunStandardScaler
        result.midResult = self.midResult
        result.predict = self.predict
        result.distributionInfo = self.distributionInfo
        return result
        
    def SelectTopN(self,N):
        id = 0
        for item in self.data:
            if id >= N:
                return
            if self.labeled:
                print(item,"Label:",self.predict[id])
            else:
                print(item)
            id += 1

    def DBSCAN(self,eps=3.4, min_samples=200):#(3,100) is good
        result = Data()
        result.data = self.data[:]
        result.dimensions = self.dimensions[:]
        clf = sklearn.cluster.DBSCAN(eps=eps,min_samples=min_samples)
        result.predict = clf.fit_predict(result.data)
        self.midResult = clf
        result.labeled = True
        return result

    def KMeans(self,K):
        result = Data()
        result.data = self.data[:]
        result.dimensions = self.dimensions[:]
        clf = sklearn.cluster.KMeans(n_clusters=K)
        result.predict = clf.fit_predict(result.data)
        self.midResult = clf
        result.labeled = True
        return result

    def MiniBatchKMeans(self,K):
        result = Data()
        result.data = self.data[:]
        result.dimensions = self.dimensions[:]
        clf = sklearn.cluster.MiniBatchKMeans(n_clusters=K)
        result.predict = clf.fit_predict(result.data)
        self.midResult = clf
        result.labeled = True
        return result
    
    def myKMeans(self,K):
        result = Data()
        result.data = self.data[:]
        result.dimensions = self.dimensions[:]
        
        dataSet =numpy.array(result.data)
        m = numpy.shape(dataSet)[0] 
        n=len(result.dimensions)
        #print(m,n)
        centroids = numpy.zeros((K,n))
        a=numpy.random.choice(a=m, size=K, replace=False, p=None)
        for i in range(K):
            centroids[i,:] =dataSet[a[i],:]
        clusterChange = True 
        clusterAssment = numpy.mat(numpy.zeros((m,2)))
        while clusterChange:
            clusterChange = False  
            for i in range(m):
                minDist=100000.0
                minIndex=-1
                
                #遍历质心
                for j in range(K):
                    distance=numpy.fabs( centroids[j,:] - dataSet[i,: ]).sum(axis=0)
                    if distance< minDist:
                        minDist=distance
                        minIndex=j
                if clusterAssment[i,0] != minIndex:
                    clusterChange = True
                    clusterAssment[i,:] = minIndex,minDist
            for j in range(K):
                j_points= dataSet[ numpy.nonzero(clusterAssment[:,0].A == j) [0] ]  
                centroids[j,:] = numpy.mean(j_points,axis=0) 
        result.predict = numpy.array([0]*len(result.data))
        for i in range(m):
             result.predict[i]=int(clusterAssment[:,0][i][0])
        result.labeled = True
        return result


    def pca(self,N):
        result = Data()
        pca = PCA(n_components=N)
        dataSet = numpy.array(self.data)
        #print(dataSet.shape)
        pcaData = pca.fit_transform(dataSet)
        print('Size to {}'.format(pcaData.shape))
        result.data = pcaData.tolist()
        result.dimensions = list(map(lambda x: '#{0}'.format(x), range(0,len(result.data[0]))))
        return result
        

    # We use t-SNE show high dimensions data...
    def Draw(self):
        seaborn.set_style('darkgrid')
        seaborn.set_palette('muted')
        seaborn.set_context("notebook", font_scale=1.5,
                        rc={"lines.linewidth": 2.5})
        colorCount = max(self.predict)+1 if self.labeled else 1
        def scatter(x, colors):
            # We choose a color palette with seaborn.
            palette = numpy.array(seaborn.color_palette("hls", colorCount))

            # We create a scatter plot.
            f = matplotlib.pyplot.figure(figsize=(8, 8))
            ax = matplotlib.pyplot.subplot(aspect='equal')
            sc = ax.scatter(x[:,0], x[:,1], lw=0, s=20,
                                c=list(map( # This is to make the unclassified node black!
                                        lambda x: palette[x] if x >= 0 else [0, 0, 0, 1],
                                        colors.astype(numpy.int))))
                            # c=map(Unclassified2Black,palette[colors.astype(numpy.int)]))
            matplotlib.pyplot.xlim(-25, 25)
            matplotlib.pyplot.ylim(-25, 25)
            ax.axis('off')
            ax.axis('tight')

            # We add the labels for each digit.
            txts = []
            for i in range(colorCount):
                # Position of each label.
                xtext, ytext = numpy.median(x[colors == i, :], axis=0)
                txt = ax.text(xtext, ytext, str(i), fontsize=34)
                txt.set_path_effects([
                    matplotlib.patheffects.Stroke(linewidth=5, foreground="w"),
                    matplotlib.patheffects.Normal()])
                txts.append(txt)
            return f, ax, sc, txts

        RS = 123456
        digits_proj = sklearn.manifold.TSNE(random_state=RS).fit_transform(self.data)
        scatter(digits_proj, self.predict if self.labeled else numpy.array([0]*len(self.data)))

        foo_fig = matplotlib.pyplot.gcf()
        foo_fig.savefig('demo.eps', format='eps', dpi=1000)
        matplotlib.pyplot.show()

    def ShowLabelInfo(self,output=True):
        if not self.labeled:
            print('No Labels yet.')
        labelCount = max(self.predict) + 1
        distribution = list()
        for i in range(labelCount):
            distribution.append(list())
        for i in range(len(self.data)):
            if self.predict[i]<0:
                continue
            distribution[self.predict[i]].append(self.data[i])
        self.distributionInfo = dict()
        self.distributionInfo['Num'] = list(map(len,distribution))
        self.distributionInfo['Max'] = list(map(lambda x: list(map(max,list(zip(*x)))),distribution))
        self.distributionInfo['Min'] = list(map(lambda x: list(map(min,list(zip(*x)))),distribution))
        self.distributionInfo['Avg'] = list(map(lambda x: list(map(numpy.mean,list(zip(*x)))),distribution))
        self.distributionInfo['Med'] = list(map(lambda x: list(map(numpy.median,list(zip(*x)))),distribution))
        self.distributionInfo['Std'] = list(map(lambda x: list(map(numpy.std,list(zip(*x)))),distribution))
        floatRound = lambda Xlist: list(map(lambda x: round(1.0*x,2), Xlist))
        for (k,v) in self.distributionInfo.items():
            if k=='Num':
                continue
            self.distributionInfo[k] = list(map(floatRound,v))
        if output:
            print('There are {0} labels.'.format(labelCount))
            for i in range(labelCount):
                print('Label {0}: Num:{1},\n Min:{2},\n Max:{3},\n Avg:{4},\n Med:{5},\n Std:{6}\n\n'.format(i + 1, self.distributionInfo['Num'][i], self.distributionInfo['Min'][i],
                    self.distributionInfo['Max'][i], self.distributionInfo['Avg'][i], self.distributionInfo['Med'][i], self.distributionInfo['Std'][i]))

if __name__ == "__main__":
    test = Data()
    test.RunNormalize = False
    test.ReadData('data.csv')
    # test.Draw()
    test.SelectTopN(10)
    #降到5维
    test = test.pca(.95)
    # result = test.KMeans(6)
    result = test.myKMeans(6)
    result.SelectTopN(10)
    result.ShowLabelInfo()
    result.Draw()
