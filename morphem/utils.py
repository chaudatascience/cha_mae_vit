import numpy as np
import faiss

########################################################
## KNN Classifier
########################################################

class FaissKNeighbors:
    def __init__(self, k, use_gpu: bool, metric: str):
        """
        @param k: number of neighbors
        @param use_gpu: True to use GPU, False to use CPU
        @param metric: "l2", "cosine"
        """
        self.index = None
        self.y = None
        self.k = k
        self.gpu = faiss.StandardGpuResources() if use_gpu else None
        self.metric = metric
        assert metric in ['l2', 'cosine'], f"{metric} is not a valid metric. Choose from ['l2', 'cosine']"

    def fit(self, X, y):
        if self.metric == 'cosine':
            # L2-normalize the vectors before computing dot product
            X = X / np.linalg.norm(X, axis=-1, keepdims=True)
            if self.gpu:
                self.index = faiss.GpuIndexFlatIP(self.gpu, X.shape[1])
            else:
                self.index = faiss.IndexFlatIP(X.shape[1])
        else:  ## "l2"
            if self.gpu:
                self.index = faiss.GpuIndexFlatL2(self.gpu, X.shape[1])
            else:
                self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X_test):
        if self.metric == 'cosine':
            # L2-normalize the vectors before computing dot product
            X_test = X_test / np.linalg.norm(X_test, axis=-1, keepdims=True)

        distances, indices = self.index.search(X_test.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions
