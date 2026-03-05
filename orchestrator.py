import json
import time
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import numpy as np

from src.extractor.sdk_extractor import SDKFeatureExtractor
from src.algorithm.factory import AlgorithmFactory
from src.evaluation.metrics_evaluator import MetricsEvaluator


class PipelineOrchestrator:
    """
    SDK 聚类自动化实验编排中枢。
    负责维持特征矩阵的持久化缓存，并根据传入的实验配置批量执行调度。
    """

    def __init__(self, raw_data_dir: str, cache_file: str):
        self.raw_data_dir = raw_data_dir
        self.cache_file = Path(cache_file)
        self.base_matrix = None
        self.sample_info = None
        self.evaluator = None

        self._initialize_base_data()

    def _initialize_base_data(self):
        if self.cache_file.exists():
            print(f"命中本地特征缓存，正在从 {self.cache_file} 加载高维矩阵...")
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.base_matrix = np.array(data['feature_matrix'])
            self.sample_info = data['sample_info']
            print(f"缓存加载完毕。矩阵维度: {self.base_matrix.shape}")
        else:
            print("未探测到特征缓存，正在激活基础特征提取流水线...")
            extractor = SDKFeatureExtractor()
            extractor.load_data(self.raw_data_dir)
            self.base_matrix, self.sample_info = extractor.extract_high_dim_matrix()

            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'feature_matrix': self.base_matrix.tolist(),
                    'sample_info': self.sample_info
                }, f, ensure_ascii=False)
            print("底层特征提取已执行完毕并落盘缓存。")

        self.evaluator = MetricsEvaluator(self.sample_info)

    def run_experiments(self, experiments_config: List[Dict]):
        print("\n" + "=" * 80)
        print("启动自动化算法对比实验流水线")
        print("=" * 80)

        results = []
        dim_cache = {}

        for idx, exp in enumerate(experiments_config):
            name = exp['name']
            print(f"\n[实验组 {idx + 1}/{len(experiments_config)}] 正在执行参数配置: {name}")
            start_time = time.time()

            try:
                # 步骤一：特征流形映射 (降维)
                dim_key = f"{exp['dim_method']}_{exp['dim_target']}"
                if dim_key not in dim_cache:
                    reducer = AlgorithmFactory.create_reducer(exp['dim_method'], exp['dim_target'])
                    dim_cache[dim_key] = reducer.fit_transform(self.base_matrix)
                current_matrix = dim_cache[dim_key]

                # 步骤二：密度或空间聚类
                clusterer = AlgorithmFactory.create_clusterer(exp['cluster_algo'], exp['cluster_params'])
                labels = clusterer.fit_predict(current_matrix)

                # 步骤三：业务形态评估
                metrics = self.evaluator.evaluate(labels, current_matrix)
                cost_time = time.time() - start_time

                row = {'Experiment_Name': name, 'Time_Cost_Sec': round(cost_time, 2)}
                row.update(metrics)
                results.append(row)

                # 步骤四：输出当前配置的可视化 JSON
                self._export_clusters(name, labels)

            except Exception as e:
                print(f"执行当前配置时遭遇异常: {e}")
                import traceback
                traceback.print_exc()

        self._generate_report(results)

    def _export_clusters(self, experiment_name: str, labels: np.ndarray):
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', experiment_name)
        output_path = Path(f"./data/processed/cluster_result_{safe_name}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cluster_map = defaultdict(list)
        for i, label in enumerate(labels):
            cid = f"Cluster_{label}" if label != -1 else "Noise"
            info = self.sample_info[i]
            ver = info.get('version', '')
            full_name = f"{info['coordinateName']}@{ver}" if ver else info['coordinateName']
            cluster_map[cid].append(full_name)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_map, f, ensure_ascii=False, indent=2)

    def _generate_report(self, results: List[Dict]):
        df = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("流水线执行完毕，生成终极对比评估报告")
        print("=" * 80)
        print(df.to_string(index=False))

        report_path = Path("./data/processed/experiment_tuning_report.csv")
        df.to_csv(report_path, index=False, encoding='utf-8-sig')
        print(f"\n对比报告已持久化至: {report_path}")