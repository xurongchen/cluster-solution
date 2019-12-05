import os,re
import sklearn.cluster
import sklearn.manifold

import numpy
import matplotlib.pyplot
import matplotlib.patheffects
import seaborn
class Data:
    data = []
    dimensions = []
    labeled = False
    def ReadData(self, filePath):
        if not os.path.isfile(filePath):
            raise FileNotFoundError('Data file not found!')
        fopen = open(filePath)
        isTitle = True
        self.data = []
        for line in fopen.readlines():
            if isTitle:
                self.dimensions = re.findall(r'"(.*?)"',line)
                isTitle = False
            else:
                self.data.append(tuple(map(int,line.strip().split(','))))
        fopen.close()

    def SelectTopN(self,N):
        id = 0
        for item in self.data:
            if id >= N:
                return
            if self.labeled:
                print(item,"Label:",self.predict[id])
            else:
                print(item)
            id += 1

    def KMeans(self,K):
        result = Data()
        result.data = self.data[:]
        result.dimensions = self.dimensions[:]
        clf = sklearn.cluster.KMeans(n_clusters=K)
        result.predict = clf.fit_predict(result.data)
        result.labeled = True
        return result

    #TODO:PCA

    # We use t-SNE show high dimensions data...
    def Draw(self):
        seaborn.set_style('darkgrid')
        seaborn.set_palette('muted')
        seaborn.set_context("notebook", font_scale=1.5,
                        rc={"lines.linewidth": 2.5})
        colorCount = max(self.predict)+1 if self.labeled else 1
        def scatter(x, colors):
            # We choose a color palette with seaborn.
            palette = numpy.array(seaborn.color_palette("hls", colorCount))

            # We create a scatter plot.
            f = matplotlib.pyplot.figure(figsize=(8, 8))
            ax = matplotlib.pyplot.subplot(aspect='equal')
            sc = ax.scatter(x[:,0], x[:,1], lw=0, s=20,
                            c=palette[colors.astype(numpy.int)])
            matplotlib.pyplot.xlim(-25, 25)
            matplotlib.pyplot.ylim(-25, 25)
            ax.axis('off')
            ax.axis('tight')

            # We add the labels for each digit.
            txts = []
            for i in range(colorCount):
                # Position of each label.
                xtext, ytext = numpy.median(x[colors == i, :], axis=0)
                txt = ax.text(xtext, ytext, str(i), fontsize=34)
                txt.set_path_effects([
                    matplotlib.patheffects.Stroke(linewidth=5, foreground="w"),
                    matplotlib.patheffects.Normal()])
                txts.append(txt)

            return f, ax, sc, txts
        RS = 123456
        digits_proj = sklearn.manifold.TSNE(random_state=RS).fit_transform(self.data)
        scatter(digits_proj, self.predict if self.labeled else 0)

        foo_fig = matplotlib.pyplot.gcf()
        foo_fig.savefig('demo.eps', format='eps', dpi=1000)
        matplotlib.pyplot.show()

if __name__ == "__main__":
    test = Data()
    test.ReadData('data.csv')
    test.SelectTopN(10)
    result = test.KMeans(20)
    result.SelectTopN(10)
    result.Draw()
