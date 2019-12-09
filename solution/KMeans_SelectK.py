from Data import Data
import matplotlib.pyplot


SSE = [] # sum of the squared errors
TestUpperBound = 15
for k in range(1,TestUpperBound):
    print('Now @ k = {0}'.format(k))
    testK = Data()
    testK.ReadData('data.csv')
    result = testK.KMeans(k)
    SSE.append(testK.midResult.inertia_)

x = range(1,TestUpperBound)
matplotlib.pyplot.figure(figsize=(5,5))
matplotlib.pyplot.xlabel('k')
matplotlib.pyplot.ylabel('SSE')  
matplotlib.pyplot.plot(x,SSE,'o-')  
matplotlib.pyplot.show()

