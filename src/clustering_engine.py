import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not installed. HDBSCAN clustering will be unavailable.")


class SDKClusteringEngine:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.feature_matrix = None
        self.sample_info = None
        self.labels = None
        self.model = None
        self.cluster_centers = None
        self.cluster_stats = None
        
    def _default_config(self) -> Dict:
        return {
            'algorithm': 'dbscan',
            'dbscan': {
                'eps': 0.5,
                'min_samples': 5,
                'metric': 'euclidean',
                'n_jobs': -1
            },
            'kmeans': {
                'n_clusters': 8,
                'init': 'k-means++',
                'n_init': 10,
                'max_iter': 300,
                'random_state': 42
            },
            'hdbscan': {
                'min_cluster_size': 5,
                'min_samples': None,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom'
            },
            'agglomerative': {
                'n_clusters': 8,
                'linkage': 'ward',
                'affinity': 'euclidean'
            },
            'visualization': {
                'pca_components': 2,
                'figsize': (12, 8)
            }
        }
    
    def load_features(self, feature_path: str) -> np.ndarray:
        with open(feature_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.feature_matrix = np.array(data['feature_matrix'])
        self.sample_info = data.get('sample_info', [])
        
        print(f"Loaded features with shape: {self.feature_matrix.shape}")
        print(f"Number of samples: {len(self.sample_info)}")
        
        return self.feature_matrix
    
    def fit_dbscan(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        if X is None:
            X = self.feature_matrix
        
        params = self.config['dbscan']
        print(f"\nRunning DBSCAN with eps={params['eps']}, min_samples={params['min_samples']}")
        
        self.model = DBSCAN(
            eps=params['eps'],
            min_samples=params['min_samples'],
            metric=params['metric'],
            n_jobs=params['n_jobs']
        )
        self.labels = self.model.fit_predict(X)
        
        return self.labels
    
    def fit_kmeans(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        if X is None:
            X = self.feature_matrix
        
        params = self.config['kmeans']
        n_samples = X.shape[0]
        
        # 确保聚类数不超过样本数
        n_clusters = min(params['n_clusters'], n_samples)
        if n_clusters < params['n_clusters']:
            print(f"Warning: Reducing n_clusters from {params['n_clusters']} to {n_clusters} due to insufficient samples")
        
        print(f"\nRunning K-Means with n_clusters={n_clusters}")
        
        self.model = KMeans(
            n_clusters=n_clusters,
            init=params['init'],
            n_init=params['n_init'],
            max_iter=params['max_iter'],
            random_state=params['random_state']
        )
        self.labels = self.model.fit_predict(X)
        self.cluster_centers = self.model.cluster_centers_
        
        return self.labels
    
    def fit_hdbscan(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
        
        if X is None:
            X = self.feature_matrix
        
        params = self.config['hdbscan']
        print(f"\nRunning HDBSCAN with min_cluster_size={params['min_cluster_size']}")
        
        self.model = HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            metric=params['metric'],
            cluster_selection_method=params['cluster_selection_method']
        )
        self.labels = self.model.fit_predict(X)
        
        return self.labels
    
    def fit_agglomerative(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        if X is None:
            X = self.feature_matrix
        
        params = self.config['agglomerative']
        n_samples = X.shape[0]
        
        # 确保聚类数不超过样本数
        n_clusters = min(params['n_clusters'], n_samples)
        if n_clusters < params['n_clusters']:
            print(f"Warning: Reducing n_clusters from {params['n_clusters']} to {n_clusters} due to insufficient samples")
        
        print(f"\nRunning Agglomerative Clustering with n_clusters={n_clusters}")
        
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=params['linkage'],
            affinity=params['affinity']
        )
        self.labels = self.model.fit_predict(X)
        
        return self.labels
    
    def fit(self, algorithm: Optional[str] = None, X: Optional[np.ndarray] = None) -> np.ndarray:
        algo = algorithm or self.config['algorithm']
        
        if algo == 'dbscan':
            self.labels = self.fit_dbscan(X)
        elif algo == 'kmeans':
            self.labels = self.fit_kmeans(X)
        elif algo == 'hdbscan':
            self.labels = self.fit_hdbscan(X)
        elif algo == 'agglomerative':
            self.labels = self.fit_agglomerative(X)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        
        self._compute_cluster_stats()
        self._print_cluster_summary()
        
        return self.labels
    
    def predict_cluster(self, sample: np.ndarray) -> int:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(sample.reshape(1, -1))[0]
        else:
            centers = self.cluster_centers
            if centers is not None:
                distances = np.linalg.norm(centers - sample, axis=1)
                return np.argmin(distances)
            else:
                raise NotImplementedError("Predict not supported for this algorithm")
    
    def add_new_sample(self, sample: np.ndarray) -> int:
        if self.feature_matrix is None:
            raise RuntimeError("No feature matrix loaded.")
        
        cluster_id = self.predict_cluster(sample)
        
        self.feature_matrix = np.vstack([self.feature_matrix, sample])
        self.labels = np.append(self.labels, cluster_id)
        
        print(f"New sample assigned to cluster {cluster_id}")
        
        return cluster_id
    
    def _compute_cluster_stats(self):
        unique_labels = np.unique(self.labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(self.labels).count(-1)
        
        self.cluster_stats = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_sizes': {}
        }
        
        for label in unique_labels:
            if label == -1:
                continue
            self.cluster_stats['cluster_sizes'][int(label)] = list(self.labels).count(label)
    
    def _print_cluster_summary(self):
        if self.cluster_stats is None:
            return
        
        print("\n" + "="*50)
        print("CLUSTERING SUMMARY")
        print("="*50)
        print(f"Number of clusters: {self.cluster_stats['n_clusters']}")
        print(f"Number of noise points: {self.cluster_stats['n_noise']}")
        
        if self.cluster_stats['cluster_sizes']:
            print("\nCluster sizes:")
            for cluster_id, size in sorted(self.cluster_stats['cluster_sizes'].items()):
                print(f"  Cluster {cluster_id}: {size} samples")
        
        print("="*50)
    
    def evaluate(self, X: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None) -> Dict:
        if X is None:
            X = self.feature_matrix
        if labels is None:
            labels = self.labels
        
        metrics = {}
        
        if len(set(labels)) > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(X, labels)
            except:
                pass
            
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
            except:
                pass
            
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            except:
                pass
        
        return metrics
    
    def save_results(self, output_path: str):
        if self.labels is None or self.sample_info is None:
            raise RuntimeError("No clustering results to save.")
        
        results = {
            'algorithm': self.config['algorithm'],
            'algorithm_params': self.config.get(self.config['algorithm'], {}),
            'cluster_labels': self.labels.tolist(),
            'cluster_stats': self.cluster_stats,
            'evaluation_metrics': self.evaluate(),
            'samples': []
        }
        
        for i, sample_info in enumerate(self.sample_info):
            sample_data = sample_info.copy()
            sample_data['cluster_id'] = int(self.labels[i])
            results['samples'].append(sample_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nClustering results saved to {output_path}")
    
    def visualize_clusters(self, X: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None, 
                       save_path: Optional[str] = None):
        if X is None:
            X = self.feature_matrix
        if labels is None:
            labels = self.labels
        
        if labels is None:
            raise RuntimeError("No labels to visualize.")
        
        viz_config = self.config['visualization']
        n_components = viz_config['pca_components']
        
        if X.shape[1] > n_components:
            pca = PCA(n_components=n_components)
            X_2d = pca.fit_transform(X)
            print(f"Reduced to {n_components}D for visualization (explained variance: {pca.explained_variance_ratio_.sum():.2%})")
        else:
            X_2d = X
        
        fig, ax = plt.subplots(figsize=viz_config['figsize'])
        
        unique_labels = set(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, (label, color) in enumerate(zip(sorted(unique_labels), colors)):
            if label == -1:
                cluster_name = 'Noise'
                marker = 'x'
                alpha = 0.5
            else:
                cluster_name = f'Cluster {label}'
                marker = 'o'
                alpha = 0.7
            
            mask = labels == label
            if X_2d.shape[1] >= 2:
                ax.scatter(
                    X_2d[mask, 0],
                    X_2d[mask, 1],
                    c=[color],
                    label=cluster_name,
                    marker=marker,
                    alpha=alpha,
                    s=100
                )
            else:
                ax.scatter(
                    X_2d[mask, 0],
                    np.zeros(np.sum(mask)),
                    c=[color],
                    label=cluster_name,
                    marker=marker,
                    alpha=alpha,
                    s=100
                )
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2' if X_2d.shape[1] >= 2 else '')
        ax.set_title(f'Clustering Results ({self.config["algorithm"].upper()})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_algorithms(self, X: Optional[np.ndarray] = None) -> Dict:
        if X is None:
            X = self.feature_matrix
        
        results = {}
        algorithms = ['dbscan', 'kmeans', 'agglomerative']
        
        if HDBSCAN_AVAILABLE:
            algorithms.append('hdbscan')
        
        for algo in algorithms:
            print(f"\n{'='*50}")
            print(f"Testing: {algo.upper()}")
            print('='*50)
            
            self.config['algorithm'] = algo
            self.fit(algo, X)
            
            metrics = self.evaluate(X, self.labels)
            results[algo] = {
                'n_clusters': self.cluster_stats['n_clusters'],
                'n_noise': self.cluster_stats['n_noise'],
                'metrics': metrics
            }
        
        return results
    
    def print_comparison(self, comparison_results: Dict):
        print("\n" + "="*80)
        print("ALGORITHM COMPARISON")
        print("="*80)
        print(f"{'Algorithm':<15} {'Clusters':<10} {'Noise':<10} {'Silhouette':<12} {'DB Index':<12} {'CH Index':<12}")
        print("-"*80)
        
        for algo, result in comparison_results.items():
            metrics = result['metrics']
            silhouette = f"{metrics.get('silhouette_score', 0):.4f}" if 'silhouette_score' in metrics else "N/A"
            db = f"{metrics.get('davies_bouldin_score', 0):.4f}" if 'davies_bouldin_score' in metrics else "N/A"
            ch = f"{metrics.get('calinski_harabasz_score', 0):.4f}" if 'calinski_harabasz_score' in metrics else "N/A"
            
            print(f"{algo:<15} {result['n_clusters']:<10} {result['n_noise']:<10} {silhouette:<12} {db:<12} {ch:<12}")
        
        print("="*80)


def main():
    engine = SDKClusteringEngine()
    
    feature_path = '/Users/chengqihan/AICode/sdk_clustering/data/processed/processed_features.json'
    output_path = '/Users/chengqihan/AICode/sdk_clustering/data/processed/clustering_result.json'
    viz_path = '/Users/chengqihan/AICode/sdk_clustering/reports/clustering_visualization.png'
    
    try:
        engine.load_features(feature_path)
        
        print("\n" + "="*50)
        print("ALGORITHM COMPARISON")
        print("="*50)
        
        comparison_results = engine.compare_algorithms()
        engine.print_comparison(comparison_results)
        
        print("\n" + "="*50)
        print("SELECTING BEST ALGORITHM")
        print("="*50)
        
        best_algo = max(comparison_results.items(), 
                      key=lambda x: x[1]['metrics'].get('silhouette_score', -1))
        print(f"\nBest algorithm: {best_algo[0].upper()}")
        print(f"Silhouette score: {best_algo[1]['metrics'].get('silhouette_score', 0):.4f}")
        
        engine.config['algorithm'] = best_algo[0]
        engine.fit(best_algo[0])
        
        engine.save_results(output_path)
        engine.visualize_clusters(save_path=viz_path)
        
    except FileNotFoundError:
        print(f"\n特征文件未找到: {feature_path}")
        print(f"请先运行特征工程: python src/feature_engineer.py")
    except Exception as e:
        print(f"\n处理聚类时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
