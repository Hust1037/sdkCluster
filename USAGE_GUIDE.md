# SDK特征工程代码使用指南

## 快速开始（3步）

### 第1步：准备你的数据

将你的SDK数据文件放到以下位置：

```
/Users/chengqihan/AICode/sdk_clustering/data/raw/samples.json
```

**支持的数据格式：**

**格式1：单个SDK样本**
```json
{
  "coordinateName": "@abner/log",
  "version": "1.0.3",
  "codeTlshHashes": [
    "returnundefined ",
    "@:hilog hilog Log",
    "nop stricteq add2...",
    "T1A521981D0779E0E...",
    ...
  ]
}
```

**格式2：多个SDK样本（数组）**
```json
[
  {
    "coordinateName": "@abner/log",
    "version": "1.0.3",
    "codeTlshHashes": [...]
  },
  {
    "coordinateName": "@other/sdk",
    "version": "2.0.0",
    "codeTlshHashes": [...]
  }
]
```

**格式3：JSONL（每行一个样本）**
```json
{"coordinateName": "@abner/log", "version": "1.0.3", "codeTlshHashes": [...]}
{"coordinateName": "@other/sdk", "version": "2.0.0", "codeTlshHashes": [...]}
```

---

### 第2步：运行特征工程

打开终端，执行以下命令：

```bash
# 1. 进入项目目录
cd /Users/chengqihan/AICode/sdk_clustering

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 运行特征工程
python src/feature_engineer.py
```

**运行后你会看到：**
```
Loading data...
Loaded 100 samples

Extracting features...
Extracting import/export features...
Extracting opcode features...
Extracting TLSH features...
Import/Export features shape: (100, 45)
Opcode features shape: (100, 256)
TLSH features shape: (100, 70)

Fusing features...
Fused features shape: (100, 371)

Applying scaling...

Applying PCA (if configured)...

Final feature matrix shape: (100, 371)
Processed data saved to /Users/chengqihan/AICode/sdk_clustering/data/processed/processed_features.json

==================================================
DATA PROCESSING REPORT
==================================================
{
  "total_samples": 100,
  "feature_matrix_shape": [100, 371],
  "sample_statistics": {
    "with_tls": 45,
    "with_import_export": 89,
    "with_opcode": 97,
    "avg_sequences": 18.5,
    "max_sequences": 32,
    "min_sequences": 5
  }
}
```

---

### 第3步：使用处理后的特征

处理后的特征保存在：
```
/Users/chengqihan/AICode/sdk_clustering/data/processed/processed_features.json
```

**如何在Python中使用：**

```python
import json
import numpy as np
from sklearn.cluster import DBSCAN

# 1. 加载特征矩阵
with open('data/processed/processed_features.json', 'r') as f:
    data = json.load(f)

# 2. 提取特征矩阵
feature_matrix = np.array(data['feature_matrix'])
print(f"特征矩阵形状: {feature_matrix.shape}")  # (样本数, 特征数)

# 3. 获取样本信息
sample_info = data['sample_info']
print(f"第一个样本: {sample_info[0]}")

# 4. 使用特征进行聚类
clustering = DBSCAN(eps=0.5, min_samples=5)
labels = clustering.fit_predict(feature_matrix)
print(f"聚类标签: {labels}")
```

---

## 作为Python模块使用

### 方式1：直接使用主函数

```python
from src.feature_engineer import SDKFeatureEngineer

# 创建特征工程器
engineer = SDKFeatureEngineer()

# 处理数据（一步到位）
feature_matrix = engineer.process('data/raw/samples.json')

# 保存结果
engineer.save_processed_data('data/processed/my_features.json')

# 生成报告
report = engineer.generate_report()
print(report)
```

### 方式2：自定义配置

```python
from src.feature_engineer import SDKFeatureEngineer

# 自定义配置
config = {
    'max_features': 3000,        # 增加TF-IDF特征数
    'ngram_range': (1, 3),       # 使用1-3 gram
    'min_df': 3,                  # 最小文档频率
    'max_df': 0.9,               # 最大文档频率
    'pca_components': 100,         # 降维到100维
    'scaler': 'minmax',           # 使用MinMaxScaler
    'tlsh_weight': 0.2,           # TLSH权重20%
    'opcode_weight': 0.5,          # 操作码权重50%
    'import_weight': 0.3           # 导入导出权重30%
}

engineer = SDKFeatureEngineer(config)
feature_matrix = engineer.process('data/raw/samples.json')
```

### 方式3：逐步处理

```python
from src.feature_engineer import SDKFeatureEngineer

engineer = SDKFeatureEngineer()

# 1. 加载数据
data = engineer.load_data('data/raw/samples.json')

# 2. 提取特征
features_dict = engineer.extract_all_features()

# 3. 融合特征
fused_features = engineer.fuse_features(features_dict)

# 4. 标准化
scaled_features = engineer.apply_scaling(fused_features)

# 5. PCA降维
reduced_features = engineer.apply_pca(scaled_features)

print(f"最终特征: {reduced_features.shape}")
```

---

## 在Jupyter Notebook中使用

### 启动Jupyter

```bash
cd /Users/chengqihan/AICode/sdk_clustering
source venv/bin/activate
jupyter notebook
```

### 在Notebook中使用

```python
import sys
sys.path.append('/Users/chengqihan/AICode/sdk_clustering/src')

from feature_engineer import SDKFeatureEngineer
import json
import numpy as np
import matplotlib.pyplot as plt

# 创建特征工程器
engineer = SDKFeatureEngineer()

# 处理数据
feature_matrix = engineer.process('data/raw/samples.json')

# 可视化特征分布
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(feature_matrix[:, 0], bins=30)
plt.title('特征1分布')
plt.subplot(1, 2, 2)
plt.hist(feature_matrix[:, 1], bins=30)
plt.title('特征2分布')
plt.show()
```

---

## 常见问题

### Q1: 我的数据格式不对怎么办？

**A:** 确保JSON数据包含必需字段：
```json
{
  "coordinateName": "...",        // 必需：SDK名称
  "codeTlshHashes": [...]       // 必需：序列列表
  "version": "..."               // 可选：版本号
}
```

### Q2: 我想处理多个文件怎么办？

**A:** 可以修改代码或合并数据：

```python
# 方法1：合并JSON文件
import json

data = []
for file in ['data1.json', 'data2.json', 'data3.json']:
    with open(file, 'r') as f:
        content = json.load(f)
        if isinstance(content, list):
            data.extend(content)
        else:
            data.append(content)

with open('combined.json', 'w') as f:
    json.dump(data, f)

# 方法2：批量处理
from src.feature_engineer import SDKFeatureEngineer

files = ['data1.json', 'data2.json', 'data3.json']
all_features = []

for file in files:
    engineer = SDKFeatureEngineer()
    features = engineer.process(file)
    all_features.append(features)
```

### Q3: 特征矩阵太大怎么办？

**A:** 启用PCA降维：

```python
config = {
    'pca_components': 100  # 降维到100维
}

engineer = SDKFeatureEngineer(config)
feature_matrix = engineer.process('data/raw/samples.json')
```

### Q4: 如何查看某个样本的特征？

**A:** 使用get_sample_info方法：

```python
from src.feature_engineer import SDKFeatureEngineer

engineer = SDKFeatureEngineer()
engineer.load_data('data/raw/samples.json')

# 查看第0个样本的信息
info = engineer.get_sample_info(0)
print(info)

# 输出:
# {
#   'coordinateName': '@abner/log',
#   'version': '1.0.3',
#   'import_export_count': 1,
#   'opcode_count': 15,
#   'tlsh_count': 1,
#   'total_sequences': 17
# }
```

### Q5: 如何增量添加新样本？

**A:** 加载已处理数据，处理新样本，合并：

```python
from src.feature_engineer import SDKFeatureEngineer
import json
import numpy as np

# 1. 加载已有处理数据
with open('data/processed/processed_features.json', 'r') as f:
    old_data = json.load(f)
old_features = np.array(old_data['feature_matrix'])

# 2. 处理新样本
new_engineer = SDKFeatureEngineer()
new_features = new_engineer.process('new_samples.json')

# 3. 合并特征
combined_features = np.vstack([old_features, new_features])

# 4. 保存
combined_data = old_data.copy()
combined_data['feature_matrix'] = combined_features.tolist()
combined_data['sample_info'].extend(new_engineer.data)

with open('data/processed/combined_features.json', 'w') as f:
    json.dump(combined_data, f)
```

---

## 完整示例：从零到聚类

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/Users/chengqihan/AICode/sdk_clustering/src')

from feature_engineer import SDKFeatureEngineer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import json
import numpy as np

def main():
    print("="*50)
    print("SDK聚类完整流程")
    print("="*50)
    
    # 1. 特征工程
    print("\n[1/4] 特征工程处理...")
    engineer = SDKFeatureEngineer()
    feature_matrix = engineer.process('data/raw/samples.json')
    
    # 2. 生成报告
    print("\n[2/4] 数据统计报告...")
    report = engineer.generate_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # 3. 聚类
    print("\n[3/4] 执行聚类...")
    
    # DBSCAN聚类
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(feature_matrix)
    
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    print(f"DBSCAN聚类结果:")
    print(f"  - 发现簇数: {n_clusters}")
    print(f"  - 噪声点数: {n_noise}")
    
    # 计算轮廓系数
    if n_clusters > 1:
        score = silhouette_score(feature_matrix, dbscan_labels)
        print(f"  - 轮廓系数: {score:.4f}")
    
    # 4. 保存聚类结果
    print("\n[4/4] 保存聚类结果...")
    
    result = {
        'cluster_labels': dbscan_labels.tolist(),
        'cluster_info': {
            'n_clusters': n_clusters,
            'n_noise': n_noise
        },
        'sample_info': [engineer.get_sample_info(i) for i in range(len(engineer.data))]
    }
    
    # 为每个样本添加聚类标签
    for i, sample in enumerate(result['sample_info']):
        sample['cluster_id'] = int(dbscan_labels[i])
    
    with open('data/processed/clustering_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"聚类结果已保存到: data/processed/clustering_result.json")
    print("\n" + "="*50)
    print("完成!")
    print("="*50)

if __name__ == '__main__':
    main()
```

运行这个脚本：

```bash
cd /Users/chengqihan/AICode/sdk_clustering
source venv/bin/activate
python your_script.py
```

---

## 项目文件说明

```
sdk_clustering/
├── data/
│   ├── raw/
│   │   └── samples.json              # [你放这里] 原始数据
│   └── processed/
│       ├── processed_features.json      # 输出：特征矩阵
│       └── clustering_result.json     # 输出：聚类结果
├── src/
│   └── feature_engineer.py           # 特征工程模块
├── experiments/
│   └── data_exploration.ipynb       # 数据探索notebook
├── venv/                          # Python虚拟环境
├── requirements.txt                 # 依赖包
└── README.md                       # 项目说明
```

---

## 下一步

特征工程完成后，你可以：

1. **查看数据探索分析**
   ```bash
   jupyter notebook experiments/data_exploration.ipynb
   ```

2. **尝试不同的聚类算法**
   - DBSCAN（密度聚类）
   - K-Means（K均值）
   - HDBSCAN（层次密度）
   - AgglomerativeClustering（层次聚类）

3. **调优参数**
   - 调整特征权重
   - 尝试不同的PCA维度
   - 测试不同的归一化方式

4. **可视化结果**
   - 使用PCA降维到2D/3D
   - 绘制聚类散点图
   - 分析簇的特征

需要帮助实现聚类算法吗？或者有其他问题？
