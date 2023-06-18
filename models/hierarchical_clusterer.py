import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import to_tree
import string
import itertools

class HierarchicalClusterer:
    def __init__(self, **sklearn_args):
        self.is_fit = False
        self.clusterer = AgglomerativeClustering(compute_full_tree = True,
                                                 compute_distances = True,
                                                 **sklearn_args)
        # used for generating labels
        self.chars = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    
    def fit(self, data, **fit_args):
        self.clusterer.fit(data, **fit_args)
        self.is_fit = True

    def get_linkage_matrix(self):
        """Based on this:
        https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py"""
        model = self.clusterer
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        return linkage_matrix

    def library_of_babel(self, n_symbols, length):
        """Returns a generator for all possible strings from given char set and of a 
        given length. Based on this:
        https://stackoverflow.com/questions/43119744/python-generate-all-possible-strings-of-length-n"""
        chars = self.chars[:n_symbols]
        for item in itertools.product(chars, repeat = length):
            yield "".join(item)

    def generate_labels(self, n_symbols = 8):
        assert self.is_fit, "Must fit data before generating labels"
        assert n_symbols < (26 + 26 + 10), "Too many symbols requested, can only use a-z, A-Z, 0-9"

        n_points = len(self.clusterer.labels_)
        str_length = np.ceil(np.log(n_points) / np.log(n_symbols)).astype(int)

        linkage_matrix = self.get_linkage_matrix()
        tree = to_tree(linkage_matrix)

        ordered_nodes = tree.pre_order()
        babel_gen = self.library_of_babel(n_symbols, str_length)

        labels = [babel_gen.__next__() for _ in range(len(ordered_nodes))]
        zips = zip(ordered_nodes, labels)

        labels = [label for _, label in sorted(zips)]

        return labels
    
    def label_data(self, data, n_symbols = 8):
        """The full pipeline in one method."""
        self.fit(data)
        return self.generate_labels(n_symbols)
    
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt
    iris = load_iris()
    X = iris.data  

    hc = HierarchicalClusterer()
    labels = hc.label_data(X)

    plt.scatter(X[:,0], X[:,1])
    for i, label in enumerate(labels):
        plt.annotate(label, (X[i,0], X[i,1]))



