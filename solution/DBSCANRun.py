from Data import Data
test = Data()
test.ReadData('data.csv')
result = test.DBSCAN(eps=3.4,min_samples=200)
result.ShowLabelInfo()
result.Draw()