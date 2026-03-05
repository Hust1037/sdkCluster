from abc import ABC, abstractmethod
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering


class IDimReducer(ABC):
    """降维算法规范接口"""

    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        pass


class IClusterer(ABC):
    """聚类算法规范接口"""

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        pass


class PCAReducer(IDimReducer):
    def __init__(self, n_components: int):
        self.model = PCA(n_components=n_components, random_state=42)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_transform(X)


class UMAPReducer(IDimReducer):
    def __init__(self, n_components: int):
        import umap
        self.model = umap.UMAP(n_components=n_components, min_dist=0.0, random_state=42)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_transform(X)


class IdentityReducer(IDimReducer):
    """空降维策略，直接透传原矩阵"""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return X


class HDBSCANClusterer(IClusterer):
    def __init__(self, **kwargs):
        from hdbscan import HDBSCAN
        self.model = HDBSCAN(**kwargs)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)


class KMeansClusterer(IClusterer):
    def __init__(self, **kwargs):
        self.model = KMeans(**kwargs, random_state=42)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)


class DBSCANClusterer(IClusterer):
    def __init__(self, **kwargs):
        self.model = DBSCAN(**kwargs)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)


class AgglomerativeClusterer(IClusterer):
    def __init__(self, **kwargs):
        self.model = AgglomerativeClustering(**kwargs)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)
