# SDK聚类引擎 - SDK信誉检测系统

## 项目概述

本项目是一个专门针对鸿蒙应用开发的SDK信誉检测系统，通过对SDK的导入导出序列、操作码序列和方法级模糊哈希（TLSH）进行分析，实现SDK的聚类和分类，帮助识别未知SDK的类别和潜在风险。

## 项目结构

```
sdk_clustering/
├── data/
│   ├── raw/                    # 原始数据目录
│   │   └── samples.json        # 样本数据文件
│   ├── processed/              # 处理后的数据
│   └── samples/                # 样本数据
├── src/
│   ├── algorithm/              # 算法模块
│   │   ├── factory.py          # 算法工厂
│   │   └── strategies.py       # 聚类和降维策略
│   ├── evaluation/             # 评估模块
│   │   └── metrics_evaluator.py # 评估指标
│   └── extractor/              # 特征提取模块
│       ├── sdk_extractor.py    # 原始特征提取器
│       └── sdk_extractor_optimized.py # 优化版特征提取器
├── docs/                       # 文档
│   ├── TECHNICAL_GUIDE.md      # 技术指南
│   ├── SDK_CLUSTERING_ENGINE_PRESENTATION.md # 演示文稿
│   ├── OPTIMIZATION_REPORT.md  # 优化报告
│   └── UNKNOWN_SEQUENCES_GUIDE.md # Unknown序列记录指南
├── examples/                   # 示例
│   ├── optimized_extractor_usage.py # 优化版提取器使用示例
│   └── test_unknown_sequences.py # Unknown序列测试示例
├── experiments/                # 实验
│   ├── clustering_comparison.ipynb # 聚类算法对比
│   └── data_exploration.ipynb  # 数据探索分析
├── tests/                      # 测试
│   ├── test_optimized_extractor.py # 优化版提取器测试
│   ├── test_comprehensive.py   # 综合测试套件
│   └── test_unknown_recording.py # Unknown序列记录测试
├── orchestrator.py             # 实验编排器
├── run_experiments.py          # 实验运行脚本
├── requirements.txt            # Python依赖
└── README.md                   # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
cd /Users/chengqihan/AICode/sdk_clustering
pip install -r requirements.txt
```

### 2. 准备数据

将你的JSON数据文件放到 `data/raw/` 目录下

支持的数据格式：

**格式1：单个JSON对象**
```json
{
  "coordinateName": "@abner/log",
  "version": "1.0.3",
  "codeTlshHashes": [...]
}
```

**格式2：JSON数组**
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

**格式3：JSONL（每行一个JSON对象）**
```json
{"coordinateName": "@abner/log", "version": "1.0.3", "codeTlshHashes": [...]}
{"coordinateName": "@other/sdk", "version": "2.0.0", "codeTlshHashes": [...]}
```

### 3. 运行特征提取

使用原始版本提取器：
```bash
python -c "from src.extractor.sdk_extractor import SDKFeatureExtractor; extractor = SDKFeatureExtractor(); extractor.load_data('data/raw'); extractor.extract_high_dim_matrix()"
```

使用优化版本提取器：
```bash
python -c "from src.extractor.sdk_extractor_optimized import SDKFeatureExtractorOptimized; extractor = SDKFeatureExtractorOptimized(); extractor.load_data('data/raw'); extractor.extract_high_dim_matrix()"
```

### 4. 运行实验

```bash
python run_experiments.py
```

### 5. 数据分析

启动Jupyter Notebook：
```bash
jupyter notebook experiments/data_exploration.ipynb
```

## 核心功能

### 1. 特征提取（SDKFeatureExtractor）

**主要功能：**
- **数据加载与验证**：支持多种JSON格式，自动验证数据完整性
- **序列类型检测**：智能识别导入导出、操作码、业务语义、数字和TLSH哈希序列
- **特征向量化**：
  - 导入导出序列：TF-IDF向量化
  - 操作码序列：Doc2Vec嵌入
  - 业务语义序列：TF-IDF向量化
  - 数字序列：统计特征
  - TLSH哈希：转换为数值向量
- **特征融合**：多特征加权融合，自动归一化
- **Unknown序列记录**：自动保存无法识别的序列到JSON文件

**新增功能：**
- **Unknown序列记录**：
  ```python
  extractor.save_unknown_sequences('unknown_sequences.json')
  ```
  生成包含unknown序列的详细报告，便于后续分析和优化检测逻辑

### 2. 算法模块（AlgorithmFactory）

**支持的算法：**
- **降维算法**：PCA、UMAP
- **聚类算法**：K-Means、DBSCAN、HDBSCAN、层次聚类

**使用示例：**
```python
from src.algorithm.factory import AlgorithmFactory

# 创建降维器
reducer = AlgorithmFactory.create_reducer('pca', 10)
reduced_matrix = reducer.fit_transform(feature_matrix)

# 创建聚类器
clusterer = AlgorithmFactory.create_clusterer('hdbscan', {'min_cluster_size': 2})
labels = clusterer.fit_predict(reduced_matrix)
```

### 3. 评估模块（MetricsEvaluator）

**评估指标：**
- 轮廓系数（Silhouette Score）
- 戴维斯-布尔丁指数（Davies-Bouldin Score）
- 卡林斯基-哈拉巴斯指数（Calinski-Harabasz Score）
- 聚类统计（簇数量、噪声比例等）

### 4. 实验编排（PipelineOrchestrator）

**功能：**
- 特征矩阵持久化缓存
- 批量执行多组实验配置
- 自动生成对比报告
- 导出聚类结果

**配置示例：**
```python
EXPERIMENTS_CONFIG = [
    {
        'name': '测试组_PCA线性降维_HDBSCAN',
        'dim_method': 'pca',
        'dim_target': 10,
        'cluster_algo': 'hdbscan',
        'cluster_params': {'min_cluster_size': 2, 'min_samples': 3}
    }
]
```

## 序列类型说明

### 1. 导入导出序列
- **格式**: `@:hilog hilog Log`、`@normalized:N&&&@/path/to/file`
- **处理**: TF-IDF向量化
- **用途**: 识别SDK的依赖关系和接口特征

### 2. 操作码序列
- **格式**: `nop stricteq add2 callthis2 Log`
- **处理**: Doc2Vec嵌入
- **用途**: 识别代码执行逻辑和模式

### 3. 业务语义序列
- **格式**: `_getAudioCodecInfo callthis0 _getVideoCodecInfo`
- **处理**: TF-IDF向量化
- **用途**: 识别业务逻辑和功能特征

### 4. 数字序列
- **格式**: `123456 789012 3456789`
- **处理**: 统计特征（长度、均值、方差等）
- **用途**: 识别数字模式

### 5. TLSH哈希
- **格式**: `T1A521981D0779E0E55495558CB90B20183635D8A2303FF8FD657C49559AF26BEB8B20DC`
- **处理**: 转换为数值向量
- **用途**: 检测代码相似性，容忍轻微改动

### 6. Unknown序列
- **格式**: 无法识别的序列
- **处理**: 记录到JSON文件
- **用途**: 分析检测盲点，优化检测逻辑

## 使用示例

### 基础使用

```python
from src.extractor.sdk_extractor import SDKFeatureExtractor

# 创建提取器
extractor = SDKFeatureExtractor()

# 加载数据
extractor.load_data('data/raw')

# 提取特征
feature_matrix, sample_info = extractor.extract_high_dim_matrix()

# 保存unknown序列
extractor.save_unknown_sequences('data/unknown_sequences.json')

print(f"特征矩阵维度: {feature_matrix.shape}")
print(f"样本数: {len(sample_info)}")
```

### 运行完整实验

```python
# 修改run_experiments.py中的配置
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
    }
]

# 运行实验
python run_experiments.py
```

## 配置参数

### 特征提取配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_features` | TF-IDF最大特征数 | 1000 |
| `ngram_range` | N-gram范围 | (1, 2) |
| `min_df` | 最小文档频率 | 1 |
| `max_df` | 最大文档频率 | 0.95 |
| `doc2vec_dim` | Doc2Vec嵌入维度 | 100 |
| `doc2vec_epochs` | Doc2Vec训练轮数 | 20 |

### 算法配置

| 算法 | 参数 | 说明 |
|------|------|------|
| K-Means | `n_clusters` | 聚类数量 |
| DBSCAN | `eps` | 邻域半径 |
| HDBSCAN | `min_cluster_size` | 最小簇大小 |
| HDBSCAN | `min_samples` | 最小样本数 |
| PCA | `n_components` | 降维维度 |
| UMAP | `n_components` | 降维维度 |

## 输出文件

### 1. 特征矩阵缓存
- **文件**: `data/processed/base_high_dim_cache.json`
- **内容**: 包含特征矩阵、样本信息

### 2. 聚类结果
- **文件**: `data/processed/cluster_result_*.json`
- **内容**: 按簇分组的SDK列表

### 3. 实验报告
- **文件**: `data/processed/experiment_tuning_report.csv`
- **内容**: 各实验配置的评估指标对比

### 4. Unknown序列报告
- **文件**: `data/unknown_sequences.json`
- **内容**: 无法识别的序列及其统计信息

## 技术亮点

1. **智能序列检测**：多维度序列类型识别，支持复杂格式
2. **高效特征提取**：并行处理，增量更新，内存优化
3. **灵活算法选择**：多种降维和聚类算法，支持自定义参数
4. **全面评估体系**：多指标评估，自动生成对比报告
5. **Unknown序列分析**：自动检测和记录无法识别的序列
6. **跨平台兼容**：支持不同操作系统和数据格式

## 注意事项

1. **数据质量**: 确保JSON数据格式正确，包含必需字段
2. **内存使用**: 大数据集可能需要调整 `max_features` 参数
3. **TLSH哈希**: 只有序列长度超过1024时才会生成TLSH哈希
4. **性能优化**: 对于大量数据，建议使用 `sdk_extractor_optimized.py`
5. **路径配置**: 默认使用相对路径，可在 `run_experiments.py` 中修改

## 下一步

1. **规则挖掘**：从聚类结果中提取SDK类别的规则
2. **在线预测**：将新SDK匹配到已有类簇
3. **可视化工具**：添加SDK聚类结果的可视化界面
4. **模型部署**：将系统部署为服务，支持实时分析
5. **Unknown序列分析**：基于unknown序列优化检测逻辑

## 联系信息

- **项目地址**: https://github.com/Hust1037/sdkCluster
- **更新时间**: 2026-03-05
- **版本**: 2.0.0

---

**SDK聚类引擎** - 为鸿蒙应用开发提供SDK信誉检测支持