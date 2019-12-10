from Data import Data
test = Data()
test.ReadData('data.csv')
result = test.KMeans(2)
result.ShowLabelInfo()
result.Draw()