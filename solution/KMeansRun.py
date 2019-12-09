from Data import Data
test = Data()
test.ReadData('data.csv')
result = test.KMeans(6)
result.ShowLabelInfo()
result.Draw()