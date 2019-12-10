from Data import Data
test = Data()
test.ReadData('data.csv')
test = test.pca(0.9)
result = test.MiniBatchKMeans(3)
result.ShowLabelInfo()
result.Draw()