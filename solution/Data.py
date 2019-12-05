import os,re
import sklearn.cluster

class Data:
    data = []
    dimensions = []
    labeled = False
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

    def KMeans(self,K):
        result = Data()
        result.data = self.data[:]
        result.dimensions = self.dimensions[:]
        clf = sklearn.cluster.KMeans(n_clusters=K)
        result.predict = clf.fit_predict(result.data)
        result.labeled = True
        return result

    #TODO:PCA

    #TODO:t-SNE

if __name__ == "__main__":
    test = Data()
    test.ReadData('data.csv')
    test.SelectTopN(10)
    result = test.KMeans(10)
    result.SelectTopN(10)

