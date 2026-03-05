from collections import defaultdict
from typing import List, Dict, Any
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import silhouette_score


class MetricsEvaluator:
    """
    业务评价判定器。
    通过提取样本的基础名称（不带版本号），作为局部可信的基础标签（Ground Truth），
    以此来验证聚类算法是否成功地将同一 SDK 的不同变种聚合在一起。
    """

    def __init__(self, sample_info: List[Dict]):
        self.sample_info = sample_info
        self.ground_truth = self._build_ground_truth()

    def _build_ground_truth(self) -> Dict[str, List[int]]:
        name_map = defaultdict(list)
        for i, info in enumerate(self.sample_info):
            name = info.get('coordinateName', '').split(':')[0]
            if name:
                name_map[name].append(i)

        valid_truth = {k: v for k, v in name_map.items() if len(v) >= 2}
        print(f"评估引擎就绪，检测到 {len(valid_truth)} 个存在多版本的基准 SDK 家族。")
        return valid_truth

    # === 新增方法：导出基准家族名单 ===
    def export_baseline(self, output_path: str):
        """
        将内部构建的 Ground Truth 导出为人类可读的 JSON 文件。
        你可以通过这个文件，清晰地看到系统是拿哪些同名变种作为考核基准的。
        """
        baseline_data = {}
        for base_name, indices in self.ground_truth.items():
            family_members = []
            for i in indices:
                info = self.sample_info[i]
                ver = info.get('version', '')
                full_name = f"{info['coordinateName']}@{ver}" if ver else info['coordinateName']
                family_members.append(full_name)

            # 按版本号简单排个序，方便阅读
            baseline_data[base_name] = sorted(family_members)

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(baseline_data, f, ensure_ascii=False, indent=2)
        print(f"基准 SDK 家族名单已导出至: {out_file.absolute()}")

    def evaluate(self, labels: np.ndarray, matrix: np.ndarray) -> Dict[str, Any]:
        total_fams = len(self.ground_truth)
        perfect_matches = 0

        for name, indices in self.ground_truth.items():
            assigned = [labels[i] for i in indices if labels[i] != -1]
            if assigned and len(set(assigned)) == 1:
                perfect_matches += 1

        homogeneity = perfect_matches / total_fams if total_fams > 0 else 0
        noise_rate = list(labels).count(-1) / len(labels)
        unique_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        try:
            sil_score = silhouette_score(matrix, labels)
        except Exception:
            sil_score = -1.0

        return {
            'Cluster_Count': unique_clusters,
            'Noise_Rate': f"{noise_rate:.2%}",
            'Homogeneity_Rate': f"{homogeneity:.2%}",
            'Silhouette_Score': round(sil_score, 4)
        }