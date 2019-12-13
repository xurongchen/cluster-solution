from Data import Data


def CompareTwoMethod(data, method0, method1, method1DimOp):
    data0 = data.Copy()
    data1 = data.Copy()
    data1 = method1DimOp(data1)

    data0 = method0(data0)
    data1 = method1(data1)
    
    # print(data0.__sizeof__(),data1.__sizeof__())

    XMax = max(data0.predict) + 1
    YMax = max(data1.predict) + 1
    # print('XMAX',XMax,'YMAX',YMax)
    XCount = [0] * XMax
    YCount = [0] * YMax
    XYCount = [[0] * YMax for i in range(XMax)]

    for item in list(zip(data0.predict, data1.predict)):
        if item[0]<0 and item[1]<0:
            continue
        if item[0]>=0:
            XCount[item[0]] += 1
        if item[1]>=0:
            YCount[item[1]] += 1
        if item[0]>=0 and item[1]>=0:
            XYCount[item[0]][item[1]] += 1

    # print (XCount,YCount,XYCount)
    SumXMaxJaccardDistance = 0
    for x in range(XMax):
        XMaxJaccardDistance = 0
        for y in range(YMax):
            if XMaxJaccardDistance == 1:
                break
            JaccardDistance = 1.0*XYCount[x][y]/(XCount[x]+YCount[y]-XYCount[x][y])
            if JaccardDistance > XMaxJaccardDistance:
                XMaxJaccardDistance = JaccardDistance
        SumXMaxJaccardDistance += XMaxJaccardDistance
    return SumXMaxJaccardDistance * 1.0 / XMax

test = Data()
test.ReadData('data.csv')

TaskKM = lambda k: lambda x: x.KMeans(k)
TaskMK = lambda k: lambda x: x.MiniBatchKMeans(k)
TaskDS = lambda e,m: lambda x: x.DBSCAN(eps=e,min_samples=m)


tasks = list(TaskKM(v) for v in range(2,5)) + \
            list(TaskMK(v) for v in range(2,5)) + \
                list(TaskDS(3.7417,v) for v in range(150,180,10))
tid = 0
for task in tasks:
    Best = 0
    for repeat in range(100):
        # print(repeat)
        if Best > 0.9:
            break
        def SelectQ(d):
            d.data = list(x[-28:] for x in d.data)
            return d
        value = CompareTwoMethod(test,task,task,SelectQ)
        if value > Best:
            Best = value
    print(tid,'{:.4f}'.format(Best))
    tid += 1

tid = 0
for task in tasks:
    print(tid)
    tid += 1
    for pcaR in [0.98,0.95,0.9,0.8,0.5]:
        Best = 0
        for repeat in range(100):
            if Best > 0.9:
                break
            value = CompareTwoMethod(test,task,task,lambda x: x.pca(pcaR))
            if value > Best:
                Best = value
        print(pcaR,'{:.4f}'.format(Best))

# print(CompareTwoMethod(test,lambda x: x.KMeans(2),lambda x: x.KMeans(2),lambda x: x.pca(0.95)))
# print(CompareTwoMethod(test,lambda x: x.KMeans(2),lambda x: x.KMeans(2),lambda x: x.pca(0.9)))
# print(CompareTwoMethod(test,lambda x: x.KMeans(2),lambda x: x.KMeans(2),lambda x: x.pca(0.58)))
# print(CompareTwoMethod(test,lambda x: x.KMeans(2),lambda x: x.KMeans(2),lambda x: x.pca(1)))
# K=4
# print(CompareTwoMethod(test,lambda x: x.MiniBatchKMeans(K),lambda x: x.MiniBatchKMeans(K),lambda x: x.pca(0.98)))
# print(CompareTwoMethod(test,lambda x: x.MiniBatchKMeans(K),lambda x: x.MiniBatchKMeans(K),lambda x: x.pca(0.95)))
# print(CompareTwoMethod(test,lambda x: x.MiniBatchKMeans(K),lambda x: x.MiniBatchKMeans(K),lambda x: x.pca(0.9)))
# print(CompareTwoMethod(test,lambda x: x.MiniBatchKMeans(K),lambda x: x.MiniBatchKMeans(K),lambda x: x.pca(0.8)))
# print(CompareTwoMethod(test,lambda x: x.MiniBatchKMeans(K),lambda x: x.MiniBatchKMeans(K),lambda x: x.pca(0.5)))
# print(CompareTwoMethod(test,lambda x: x.MiniBatchKMeans(2),lambda x: x.MiniBatchKMeans(2),lambda x: x.pca(1)))

# print(CompareTwoMethod(test,lambda x: x.myKMeans(2),lambda x: x.myKMeans(2),lambda x: x.pca(0.9)))
# print(CompareTwoMethod(test,lambda x: x.myKMeans(2),lambda x: x.myKMeans(2),lambda x: x))
# print(CompareTwoMethod(test,lambda x: x.KMeans(2),lambda x: x.KMeans(2),lambda x: x))

# print(CompareTwoMethod(test,lambda x: x.myKMeans(2),lambda x: x.myKMeans(2),lambda x: x.pca(2)))

# print(CompareTwoMethod(test,lambda x: x.DBSCAN(eps=3.7417,min_samples=160),lambda x: x.DBSCAN(eps=3.7417,min_samples=160),lambda x: x.pca(0.98)))
# print(CompareTwoMethod(test,lambda x: x.DBSCAN(eps=3.7417,min_samples=160),lambda x: x.DBSCAN(eps=3.7417,min_samples=160),lambda x: x.pca(0.95)))
# print(CompareTwoMethod(test,lambda x: x.DBSCAN(eps=3.7417,min_samples=160),lambda x: x.DBSCAN(eps=3.7417,min_samples=160),lambda x: x.pca(0.9)))
# print(CompareTwoMethod(test,lambda x: x.DBSCAN(eps=3.7417,min_samples=160),lambda x: x.DBSCAN(eps=3.7417,min_samples=160),lambda x: x.pca(0.8)))
# print(CompareTwoMethod(test,lambda x: x.DBSCAN(eps=3.7417,min_samples=160),lambda x: x.DBSCAN(eps=3.7417,min_samples=160),lambda x: x.pca(0.5)))
# print(CompareTwoMethod(test,lambda x: x.DBSCAN(eps=3.7417,min_samples=160),lambda x: x.DBSCAN(eps=3.7417,min_samples=160),lambda x: x.pca(0.8)))
