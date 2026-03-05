import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator import PipelineOrchestrator

# =====================================================================
# SDK 聚类调参控制面板
# 你可以在此数组中任意增加、删除或修改测试用例，流水线将自动顺次执行。
# =====================================================================
EXPERIMENTS_CONFIG = [
    {
        'name': '对照组_KMeans_强行平均聚类',
        'dim_method': 'pca',
        'dim_target': 100,
        'cluster_algo': 'kmeans',
        'cluster_params': {'n_clusters': 150}
    },
    {
        'name': '测试组_PCA线性降维_HDBSCAN',
        'dim_method': 'pca',
        'dim_target': 10,
        'cluster_algo': 'hdbscan',
        'cluster_params': {'min_cluster_size': 2, 'min_samples': 3}
    },
    {
        'name': '业务组_UMAP非线性降维_HDBSCAN',
        'dim_method': 'umap',
        'dim_target': 10,
        'cluster_algo': 'hdbscan',
        'cluster_params': {'min_cluster_size': 2, 'min_samples': 3}
    },
    {
        'name': '激进组_过滤冷门微小家族_HDBSCAN',
        'dim_method': 'umap',
        'dim_target': 15,
        'cluster_algo': 'hdbscan',
        'cluster_params': {'min_cluster_size': 5, 'min_samples': 3}
    }
]

if __name__ == "__main__":
    # 配置原始数据所在路径与缓存产物存放位置
    RAW_DATA_DIRECTORY = './data/raw'
    CACHE_FILE_PATH = './data/processed/base_high_dim_cache.json'

    # 挂载指挥中心并点火发射
    orchestrator = PipelineOrchestrator(
        raw_data_dir=RAW_DATA_DIRECTORY,
        cache_file=CACHE_FILE_PATH
    )

    orchestrator.run_experiments(EXPERIMENTS_CONFIG)