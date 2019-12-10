from Data import Data
import matplotlib.pyplot, numpy, matplotlib.pyplot

from sklearn.neighbors import NearestNeighbors

testK = Data()
testK.ReadData('data.csv')

K_Neighbors = 200
neigh = NearestNeighbors(n_neighbors=K_Neighbors+1)
nbrs = neigh.fit(numpy.array(testK.data))
distances, indices = nbrs.kneighbors(testK.data)

distances = numpy.sort(distances, axis=0)
distances = distances[:,K_Neighbors]
matplotlib.pyplot.plot(distances)
matplotlib.pyplot.show()
