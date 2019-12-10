from Data import Data
test = Data()
test.ReadData('data.csv')
# test = test.pca(0.95)
result = test.DBSCAN(eps=3.7417,min_samples=160)
result.ShowLabelInfo()
result.Draw()