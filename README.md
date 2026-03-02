# SDK聚类引擎 - 特征工程模块

## 项目结构

```
sdk_clustering/
├── data/
│   ├── raw/                    # 原始数据
│   │   └── samples.json        # 将你的JSON数据放到这里
│   ├── processed/              # 处理后的数据
│   └── samples/                # 样本数据
├── src/
│   └── feature_engineer.py     # 特征工程主模块
├── experiments/
│   └── data_exploration.ipynb  # 数据探索分析notebook
├── models/                     # 模型文件
├── reports/                    # 报告文件
└── requirements.txt           # Python依赖
```

## 快速开始

### 1. 安装依赖

```bash
cd /Users/chengqihan/AICode/sdk_clustering
pip install -r requirements.txt
```

### 2. 准备数据

将你的JSON数据文件放到 `data/raw/samples.json`

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

### 3. 运行特征工程

```bash
python src/feature_engineer.py
```

### 4. 数据探索分析

启动Jupyter Notebook：

```bash
jupyter notebook experiments/data_exploration.ipynb
```

## 功能说明

### SDKFeatureEngineer 类

主要功能：

1. **数据加载与验证**
   - 支持多种JSON格式
   - 自动验证数据完整性
   - 过滤无效样本

2. **特征提取**
   - 导入导出序列：TF-IDF向量化
   - 操作码序列：TF-IDF向量化 + N-gram
   - TLSH哈希：转换为数值特征

3. **特征融合**
   - 加权融合多特征
   - 自动归一化
   - 可选PCA降维

4. **数据保存与加载**
   - 保存处理后的特征矩阵
   - 保存样本元数据
   - 支持增量处理

### 配置参数

```python
config = {
    'max_features': 2000,           # TF-IDF最大特征数
    'ngram_range': (1, 2),          # N-gram范围
    'min_df': 2,                    # 最小文档频率
    'max_df': 0.95,                 # 最大文档频率
    'pca_components': None,         # PCA降维维度（None不降维）
    'scaler': 'standard',           # 归一化方式: 'standard'或'minmax'
    'tlsh_weight': 0.3,             # TLSH特征权重
    'opcode_weight': 0.4,           # 操作码特征权重
    'import_weight': 0.3            # 导入导出特征权重
}
```

## 输出文件

### processed_features.json

包含以下内容：

```json
{
  "feature_matrix": [[...]],          # 特征矩阵
  "sample_info": [...],               # 样本信息
  "config": {...},                    # 配置参数
  "feature_shapes": {                 # 各特征维度
    "import_export": 1000,
    "opcode": 2000,
    "tlsh": 70
  }
}
```

## 使用示例

```python
from src.feature_engineer import SDKFeatureEngineer

# 创建特征工程器
engineer = SDKFeatureEngineer()

# 处理数据
feature_matrix = engineer.process('data/raw/samples.json')

# 保存处理结果
engineer.save_processed_data('data/processed/processed_features.json')

# 生成报告
report = engineer.generate_report()
print(report)
```

## 特征说明

### 1. 导入导出序列
- **格式**: `@:hilog hilog Log`
- **处理**: TF-IDF向量化
- **用途**: 识别SDK的依赖关系和接口特征

### 2. 操作码序列
- **格式**: `nop stricteq add2 callthis2 Log mShowLogLocation...`
- **处理**: TF-IDF向量化 + N-gram (1-2)
- **用途**: 识别代码执行逻辑和模式

### 3. TLSH哈希
- **格式**: `T1A521981D0779E0E55495558CB90B20183635D8A2303FF8FD657C49559AF26BEB8B20DC`
- **处理**: 转换为数值向量
- **用途**: 检测代码相似性，容忍轻微改动

## 注意事项

1. **数据质量**: 确保JSON数据格式正确，包含必需字段
2. **内存使用**: 大数据集可能需要调整 `max_features` 参数
3. **TLSH哈希**: 只有序列长度超过1024时才会生成TLSH哈希
4. **特征权重**: 根据业务需求调整特征权重

## 下一步

特征工程完成后，可以进行：

1. **聚类算法对比**: 使用 `experiments/algorithm_comparison.ipynb`
2. **模型训练**: 训练聚类模型
3. **新样本预测**: 将新SDK匹配到已有类簇
