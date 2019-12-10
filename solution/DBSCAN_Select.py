from Data import Data
import matplotlib.pyplot, numpy, matplotlib.pyplot

from sklearn.neighbors import NearestNeighbors

testK = Data()
testK.ReadData('data.csv')

neigh = NearestNeighbors(n_neighbors=10)
nbrs = neigh.fit(numpy.array(testK.data))
distances, indices = nbrs.kneighbors(testK.data)

distances = numpy.sort(distances, axis=0)
distances = distances[:,1]
matplotlib.pyplot.plot(distances)
matplotlib.pyplot.show()
