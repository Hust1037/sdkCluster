from typing import Dict, Any, Optional
from .strategies import (
    IDimReducer, IClusterer,
    PCAReducer, UMAPReducer, IdentityReducer,
    HDBSCANClusterer, KMeansClusterer, DBSCANClusterer, AgglomerativeClusterer
)


class AlgorithmFactory:
    """
    负责对象创建的工厂类，消除上层业务代码对底层算法类的直接依赖。
    """

    # 定义支持的算法名称，用于验证和文档说明
    SUPPORTED_DIM_REDUCTION_METHODS = {'pca', 'umap'}
    SUPPORTED_CLUSTERING_ALGORITHMS = {'hdbscan', 'kmeans', 'dbscan', 'agglomerative'}

    @staticmethod
    def create_reducer(method: Optional[str], target_dim: int) -> IDimReducer:
        """
        根据指定的方法和目标维度创建降维器实例。

        Args:
            method: 降维方法的名称 ('pca', 'umap')。如果为 None 或空字符串，则返回恒等变换器。
            target_dim: 目标维度。如果小于等于0，则返回恒等变换器。

        Returns:
            一个实现了 IDimReducer 接口的实例。

        Raises:
            ValueError: 如果指定的降维方法不被支持。
        """
        if not method or target_dim <= 0:
            return IdentityReducer()

        method_lower = method.lower()
        if method_lower not in AlgorithmFactory.SUPPORTED_DIM_REDUCTION_METHODS:
            raise ValueError(
                f"系统尚未支持该降维方法: '{method}'. "
                f"支持的方法有: {list(AlgorithmFactory.SUPPORTED_DIM_REDUCTION_METHODS)}"
            )

        if method_lower == 'pca':
            return PCAReducer(n_components=target_dim)
        elif method_lower == 'umap':
            return UMAPReducer(n_components=target_dim)

    @staticmethod
    def create_clusterer(algo_name: str, params: Dict[str, Any]) -> IClusterer:
        """
        根据算法名称和参数字典创建聚类器实例。

        Args:
            algo_name: 聚类算法的名称 ('hdbscan', 'kmeans', 'dbscan', 'agglomerative')。
            params: 传递给聚类算法构造函数的关键字参数字典。

        Returns:
            一个实现了 IClusterer 接口的实例。

        Raises:
            ValueError: 如果指定的聚类算法不被支持。
        """
        algo_lower = algo_name.lower()
        if algo_lower not in AlgorithmFactory.SUPPORTED_CLUSTERING_ALGORITHMS:
            raise ValueError(
                f"系统尚未支持该聚类算法: '{algo_name}'. "
                f"支持的算法有: {list(AlgorithmFactory.SUPPORTED_CLUSTERING_ALGORITHMS)}"
            )

        if algo_lower == 'hdbscan':
            return HDBSCANClusterer(**params)
        elif algo_lower == 'kmeans':
            return KMeansClusterer(**params)
        elif algo_lower == 'dbscan':
            return DBSCANClusterer(**params)
        elif algo_lower == 'agglomerative':
            return AgglomerativeClusterer(**params)
