from Data import Data
test = Data()
test.ReadData('data.csv')
test = test.pca(0.95)
result = test.MiniBatchKMeans(2)
result.ShowLabelInfo()
result.Draw()