# SDK聚类引擎技术文档

## 目录

1. [系统概述](#系统概述)
2. [特征处理流程](#特征处理流程)
3. [特征提取详解](#特征提取详解)
4. [聚类算法原理](#聚类算法原理)
5. [评估指标说明](#评估指标说明)
6. [工作流程指南](#工作流程指南)
7. [参数调优指南](#参数调优指南)

---

## 系统概述

### 架构设计

SDK聚类引擎采用**六维特征融合**架构，结合多种特征提取技术和聚类算法，为HarmonyOS应用SDK提供精准的聚类分析。

```
┌─────────────────────────────────────────────────────────────┐
│                    SDK数据输入                          │
│         (JSON文件，7000+样本)                    │
└──────────────────┬──────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              特征工程流水线                         │
│  ┌──────────────────────────────────────────────┐      │
│  │ Coordinate Name → TF-IDF (15%)          │      │
│  ├──────────────────────────────────────────────┤      │
│  │ Business Semantic → TF-IDF (15%)       │      │
│  ├──────────────────────────────────────────────┤      │
│  │ Import/Export → TF-IDF (15%)          │      │
│  ├──────────────────────────────────────────────┤      │
│  │ Opcode Sequence → Doc2Vec (35%)        │      │
│  ├──────────────────────────────────────────────┤      │
│  │ TLSH Hash → 数值转换 (10%)            │      │
│  ├──────────────────────────────────────────────┤      │
│  │ Numeric → 统计特征 (10%)             │      │
│  └──────────────────────────────────────────────┘      │
│                    ↓                              │
│              特征融合 & 归一化                       │
│                    ↓                              │
│              可选PCA降维                            │
└──────────────────┬──────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              聚类分析引擎                           │
│  ┌──────────────────────────────────────────────┐      │
│  │ 可选降维 (PCA/UMAP)                  │      │
│  └──────────────────────────────────────────────┘      │
│                    ↓                              │
│  ┌──────────────────────────────────────────────┐      │
│  │ 聚类算法 (HDBSCAN/KMeans/DBSCAN)       │      │
│  └──────────────────────────────────────────────┘      │
│                    ↓                              │
│              聚类结果输出                            │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

1. **SDKFeatureExtractor**: 特征提取和融合
2. **AlgorithmFactory**: 算法实例工厂
3. **MetricsEvaluator**: 业务导向评估器
4. **Orchestrator**: 流程编排器

---

## 特征处理流程

### 1. 数据加载与验证

#### 输入格式支持

```python
# 支持三种格式
# 1. 单个JSON对象
{"coordinateName": "@sdk", "codeTlshHashes": [...]}

# 2. JSON数组
[{"coordinateName": "@sdk1", ...}, {"coordinateName": "@sdk2", ...}]

# 3. JSONL (每行一个JSON)
{"coordinateName": "@sdk1", ...}
{"coordinateName": "@sdk2", ...}
```

#### 数据验证流程

```python
def load_data(directory_path):
    # 1. 扫描目录，发现所有JSON/JSONL文件
    json_files = glob('*.json') + glob('*.jsonl')
    
    # 2. 并行读取文件内容
    for json_file in json_files:
        content = read_file(json_file)
        samples = parse_json(content)
        all_samples.extend(samples)
    
    # 3. 验证数据完整性
    valid_data = []
    for sample in all_samples:
        if not sample.has_required_fields():
            continue
        if sample.has_empty_hash_list():
            continue
        valid_data.append(sample)
    
    return valid_data
```

#### 验证规则

- **必需字段**: `coordinateName`, `codeTlshHashes`
- **数据类型**: `codeTlshHashes` 必须为字符串数组
- **内容检查**: 至少包含一个有效序列

### 2. 序列类型识别

#### 智能分类逻辑

```python
def detect_sequence_type(sequence):
    # 1. TLSH哈希检测
    if sequence.startswith('T1') and len(sequence) == 70:
        return 'tlsh'
    
    # 2. 导入导出序列检测
    elif ':' in sequence and sequence.count(' ') < 5:
        return 'import_export'
    
    # 3. 操作码序列检测
    elif has_opcode_keywords(sequence):
        return 'opcode'
    
    # 4. 业务语义常量检测
    elif has_business_keywords(sequence):
        return 'semantic'
    
    # 5. 数字序列检测
    elif is_numeric_sequence(sequence):
        return 'numeric'
    
    return 'unknown'
```

#### 识别关键词

| 序列类型 | 识别特征 | 示例 |
|----------|----------|------|
| **TLSH Hash** | 以'T1'开头，长度70 | `T1A521981D0779E0E55495558CB90B...` |
| **Import/Export** | 包含':'，空格少 | `@:hilog hilog Log` |
| **Opcode** | 包含指令关键词 | `nop stricteq add2 callthis2` |
| **Semantic** | 包含业务常量 | `LoadMoreLayoutStatus RequestPermission` |
| **Numeric** | 纯数字序列 | `123456789 987654321` |

### 3. 特征提取详解

#### 3.1 Coordinate Name特征 (权重15%)

**技术原理**：
```python
# 1. 提取SDK名称
name = sample['coordinateName']

# 2. N-gram分词 (3-5 grams)
tokens = extract_ngrams(name, n_min=3, n_max=5)
# 示例: "@abner/log" → ["@abn", "abner", "bner/", "ner/l", "er/log"]

# 3. TF-IDF向量化
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(3, 5),
    token_pattern=r'(?u)\b\w+\b'
)
feature_vector = vectorizer.fit_transform(all_names)
```

**业务价值**：
- 捕获SDK命名模式
- 识别同源SDK的相似命名
- 提供名称级别的相似性度量

**维度**: 最多2000维 (实际取决于不同N-gram数量)

#### 3.2 Business Semantic特征 (权重15%)

**技术原理**：
```python
# 1. 提取业务语义常量
semantic_patterns = [
    r'LoadMoreLayoutStatus',    # 布局加载
    r'RequestPermission',        # 权限请求
    r'OnDataChanged',            # 数据变更
    r'Log\.',                    # 日志操作
    r'Show\.',                   # 显示操作
    # ... 更多业务关键词
]

# 2. 匹配业务语义
semantic_features = []
for sequence in all_sequences:
    matches = []
    for pattern in semantic_patterns:
        if re.search(pattern, sequence):
            matches.append(1)
        else:
            matches.append(0)
    semantic_features.append(matches)

# 3. TF-IDF向量化
vectorizer = TfidfVectorizer(max_features=2000)
feature_matrix = vectorizer.fit_transform(semantic_features)
```

**业务价值**：
- 识别SDK的业务功能
- 捕获API使用模式
- 提供业务级别的分类线索

**维度**: 最多2000维

#### 3.3 Import/Export特征 (权重15%)

**技术原理**：
```python
# 1. 构建导入导出语料库
corpus = []
for sample in samples:
    import_export_seqs = sample['import_export_sequences']
    # 合并所有序列为文档
    document = ' '.join(import_export_seqs)
    corpus.append(document)

# 2. TF-IDF向量化 (1-2 gram)
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),  # 单词和双词组合
    token_pattern=r'(?u)\b\w+\b'
)
feature_matrix = vectorizer.fit_transform(corpus)
```

**业务价值**：
- 反映SDK的依赖关系
- 识别外部接口使用
- 揭示SDK的功能边界

**维度**: 最多2000维

#### 3.4 Opcode特征 (权重35%)

**技术原理 - Doc2Vec深度学习**：
```python
# 1. 构建训练语料库
documents = []
for sample in samples:
    opcode_sequences = sample['opcode_sequences']
    for seq in opcode_sequences:
        # 分词处理
        tokens = tokenize_sequence(seq)
        # 创建TaggedDocument
        doc = TaggedDocument(words=tokens, tags=[sample_id])
        documents.append(doc)

# 2. 训练Doc2Vec模型
model = Doc2Vec(
    vector_size=128,      # 输出向量维度
    window=5,             # 上下文窗口
    min_count=1,           # 最小词频
    epochs=20,             # 训练迭代次数
    workers=4              # 并行训练
)
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=20)

# 3. 转换序列为向量
feature_vectors = []
for seq in opcode_sequences:
    tokens = tokenize_sequence(seq)
    vector = model.infer_vector(tokens)
    feature_vectors.append(vector)
```

**Doc2Vec优势**：
- **语义理解**: 捕获指令级上下文关系
- **固定维度**: 统一128维，不受序列长度影响
- **泛化能力**: 对未见过的指令组合也能生成合理向量

**业务价值**：
- 最高权重(35%)，因为最能反映SDK的核心行为
- 识别代码执行模式
- 区分功能相似的SDK

**维度**: 固定128维

#### 3.5 TLSH Hash特征 (权重10%)

**技术原理**：
```python
# 1. 解析TLSH哈希字符串
def parse_tlsh_hash(hash_string):
    # TLSH格式: T1 + 70字符十六进制
    hex_data = hash_string[2:]  # 去掉"T1"前缀
    bytes_array = bytes.fromhex(hex_data)
    
    # 2. 转换为数值向量 (每字节4位)
    vector = np.zeros(70 * 4)  # 70字符 × 4位/字符
    for i, byte_val in enumerate(bytes_array):
        vector[i*4] = (byte_val >> 6) & 0x3    # 高2位
        vector[i*4+1] = (byte_val >> 4) & 0x3  # 次高2位
        vector[i*4+2] = (byte_val >> 2) & 0x3  # 次低2位
        vector[i*4+3] = byte_val & 0x3          # 低2位
    
    return vector

# 3. 处理多个TLSH哈希
tlsh_vectors = []
for sample in samples:
    hashes = sample['tlsh_hashes']
    if hashes:
        # 平均或加权融合多个哈希
        vectors = [parse_tlsh_hash(h) for h in hashes]
        merged_vector = np.mean(vectors, axis=0)
        tlsh_vectors.append(merged_vector)
    else:
        tlsh_vectors.append(np.zeros(280))
```

**TLSH优势**：
- **模糊匹配**: 容忍轻微代码改动
- **固定长度**: 统一280维 (70字符 × 4位/字符)
- **快速比较**: 汉明距离计算高效

**业务价值**：
- 检测代码相似性
- 识别代码复用模式
- 为版本识别提供依据

**维度**: 固定280维

#### 3.6 Numeric特征 (权重10%)

**技术原理**：
```python
# 1. 提取数字序列
numeric_sequences = extract_numeric_sequences(hashes)

# 2. 统计特征提取
numeric_features = []
for seq in numeric_sequences:
    numbers = parse_numbers(seq)
    
    # 基础统计量
    features = {
        'mean': np.mean(numbers),
        'std': np.std(numbers),
        'min': np.min(numbers),
        'max': np.max(numbers),
        'median': np.median(numbers),
        'length': len(numbers),
        'sum': np.sum(numbers),
        'range': np.max(numbers) - np.min(numbers)
    }
    numeric_features.append(features)

# 3. 归一化处理
scaler = StandardScaler()
numeric_matrix = scaler.fit_transform(numeric_features)
```

**业务价值**：
- 捕获数字模式
- 补充其他特征的盲点
- 提供量化统计信息

**维度**: 8维 (每个样本8个统计量)

### 4. 特征融合流程

#### 4.1 加权融合

```python
def fuse_features(features_dict, config):
    # 1. 提取各类型特征
    coord_feat = features_dict['coordinate']      # (n_samples, 2000)
    semantic_feat = features_dict['semantic']      # (n_samples, 2000)
    import_feat = features_dict['import_export']    # (n_samples, 2000)
    opcode_feat = features_dict['opcode']          # (n_samples, 128)
    tlsh_feat = features_dict['tlsh']                # (n_samples, 280)
    numeric_feat = features_dict['numeric']           # (n_samples, 8)

    # 2. 归一化处理
    coord_norm = MinMaxScaler().fit_transform(coord_feat)
    semantic_norm = MinMaxScaler().fit_transform(semantic_feat)
    import_norm = MinMaxScaler().fit_transform(import_feat)
    opcode_norm = MinMaxScaler().fit_transform(opcode_feat)
    tlsh_norm = MinMaxScaler().fit_transform(tlsh_feat)
    numeric_norm = StandardScaler().fit_transform(numeric_feat)

    # 3. 加权融合
    fused = np.hstack([
        coord_norm * 0.15,      # 15%权重
        semantic_norm * 0.15,    # 15%权重
        import_norm * 0.15,       # 15%权重
        opcode_norm * 0.35,       # 35%权重 (最高)
        tlsh_norm * 0.10,          # 10%权重
        numeric_norm * 0.10        # 10%权重
    ])

    return fused
```

#### 4.2 最终特征矩阵

**维度计算**：
```
总维度 = 2000 + 2000 + 2000 + 128 + 280 + 8 = 6416维
```

**7000样本的内存占用**：
```
内存 = 7000 × 6416 × 4字节 / (1024^3) ≈ 0.17 GB
```

#### 4.3 可选PCA降维

```python
# 如果启用PCA降维
if config['pca_components']:
    pca = PCA(n_components=config['pca_components'])
    reduced_features = pca.fit_transform(fused_features)
    
    # 示例: 降至100维
    # 保留的方差比例
    variance_ratio = pca.explained_variance_ratio_.sum()
    print(f"PCA降维: {6416} → {100}维, 保留方差: {variance_ratio:.2%}")
```

---

## 聚类算法原理

### 1. HDBSCAN (推荐算法)

#### 算法原理

```python
# 层次密度聚类算法
class HDBSCAN:
    def __init__(self, min_cluster_size=5, min_samples=None):
        self.min_cluster_size = min_cluster_size  # 最小聚类大小
        self.min_samples = min_samples          # 最小样本数
    
    def fit_predict(self, X):
        # 1. 构建最小生成树
        mst = build_minimum_spanning_tree(X)
        
        # 2. 层次聚类
        tree = hierarchical_clustering(mst)
        
        # 3. 确定聚类边界
        clusters = extract_clusters(tree, self.min_cluster_size)
        
        # 4. 标记噪声点
        labels = mark_noise_points(clusters, self.min_samples)
        
        return labels
```

#### 参数说明

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|---------|------|
| **min_cluster_size** | 最小聚类样本数 | 10-20 | 聚类粒度 |
| **min_samples** | 核心点最小邻域样本 | 5-10 | 噪声识别敏感度 |
| **metric** | 距离度量 | 'euclidean' | 相似性计算方式 |

#### 业务优势

1. **自动聚类数量**: 无需预先指定k值
2. **噪声处理**: 自动识别异常SDK
3. **层次结构**: 发现SDK的层次关系
4. **形状适应**: 不受球形假设限制

### 2. DBSCAN

#### 算法原理

```python
class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps              # 邻域半径
        self.min_samples = min_samples  # 最小样本数
    
    def fit_predict(self, X):
        # 1. 计算距离矩阵
        distances = compute_distance_matrix(X)
        
        # 2. 识别核心点
        core_points = find_core_points(distances, self.eps, self.min_samples)
        
        # 3. 构建聚类
        clusters = build_clusters(core_points, distances, self.eps)
        
        # 4. 标记噪声点
        labels = assign_labels(clusters, core_points)
        
        return labels
```

#### 参数说明

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|---------|------|
| **eps** | 邻域半径 | 0.3-1.0 | 聚类粒度 |
| **min_samples** | 核心点最小邻域样本 | 5-10 | 噪声识别 |

#### 业务优势

1. **噪声识别**: 强大的噪声处理能力
2. **复杂形状**: 处理非球形聚类
3. **无需k值**: 自动确定聚类数量

#### 调参挑战

- `eps`参数对结果影响大，需要多次实验
- 不同密度区域需要不同参数

### 3. K-Means

#### 算法原理

```python
class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters      # 聚类数量
        self.max_iter = max_iter        # 最大迭代次数
    
    def fit_predict(self, X):
        # 1. 随机初始化k个中心点
        centroids = initialize_centroids(X, self.n_clusters)
        
        # 2. 迭代优化
        for iteration in range(self.max_iter):
            # 分配样本到最近中心
            labels = assign_to_nearest_centroid(X, centroids)
            
            # 更新中心点
            new_centroids = update_centroids(X, labels, self.n_clusters)
            
            # 检查收敛
            if converged(centroids, new_centroids):
                break
            centroids = new_centroids
        
        return labels
```

#### 参数说明

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|---------|------|
| **n_clusters** | 聚类数量 | 根据业务需求 | 聚类粒度 |
| **max_iter** | 最大迭代次数 | 300 | 收敛速度 |

#### 业务优势

1. **计算快速**: O(n × k × i)复杂度
2. **实现简单**: 易于理解和调优

#### 业务劣势

1. **需要k值**: 必须预先知道聚类数量
2. **球形假设**: 假设聚类为球形分布
3. **噪声敏感**: 对异常点敏感

### 4. Agglomerative Clustering

#### 算法原理

```python
class AgglomerativeClustering:
    def __init__(self, n_clusters=8, linkage='ward'):
        self.n_clusters = n_clusters  # 目标聚类数
        self.linkage = linkage     # 连接方法
    
    def fit_predict(self, X):
        # 1. 计算距离矩阵
        distance_matrix = compute_distance_matrix(X)
        
        # 2. 层次合并
        linkage_matrix = linkage_method(distance_matrix, self.linkage)
        
        # 3. 切割树结构
        labels = cut_tree(linkage_matrix, self.n_clusters)
        
        return labels
```

#### 连接方法

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| **ward** | 最小化方差 | 球形聚类 |
| **complete** | 最大距离 | 紧凑聚类 |
| **average** | 平均距离 | 平衡聚类 |
| **single** | 最小距离 | 链状聚类 |

#### 业务优势

1. **层次结构**: 提供完整的层次关系
2. **无需中心**: 不依赖中心点概念

#### 业务劣势

1. **计算复杂**: O(n³)复杂度，不适合大数据集
2. **内存消耗**: 需要存储完整距离矩阵

---

## 评估指标说明

### 1. 聚类数量

```python
cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
```

**业务意义**:
- 反映发现的SDK类别数量
- 用于评估聚类的粒度
- 指导业务覆盖范围

### 2. 噪声率

```python
noise_count = list(labels).count(-1)
noise_rate = noise_count / len(labels)
```

**业务意义**:
- 反映异常SDK的比例
- 高噪声率可能表示数据质量问题
- 噪声点需要单独分析

### 3. 同质性

```python
# 基于Ground Truth计算
def compute_homogeneity(labels, ground_truth):
    perfect_matches = 0
    for family in ground_truth:
        # 检查同一SDK家族是否被分配到同一聚类
        family_labels = [labels[i] for i in ground_truth[family] if labels[i] != -1]
        if len(set(family_labels)) == 1:  # 全部分配到同一聚类
            perfect_matches += 1
    
    return perfect_matches / len(ground_truth)
```

**业务意义**:
- 衡量聚类的业务正确性
- 高同质性表示同一SDK被正确聚合
- 核心业务指标

### 4. 轮廓系数

```python
from sklearn.metrics import silhouette_score

silhouette = silhouette_score(feature_matrix, labels)
```

**计算原理**:
```
对于每个样本i:
  a(i) = 样本i到同簇其他样本的平均距离
  b(i) = 样本i到最近异簇样本的平均距离
  s(i) = (b(i) - a(i)) / max(a(i), b(i))

轮廓系数 = 所有样本s(i)的平均值
```

**取值范围**: [-1, 1]
- **接近1**: 样本与同簇相似，与异簇差异大
- **接近0**: 样本在簇边界上
- **接近-1**: 样本可能被分配到错误的簇

**业务意义**:
- 反映聚类的分离度和紧凑度
- 高轮廓系数表示聚类效果好
- 用于算法对比和参数调优

### 5. Davies-Bouldin指数

```python
from sklearn.metrics import davies_bouldin_score

db_index = davies_bouldin_score(feature_matrix, labels)
```

**计算原理**:
```
DB = (1/k) × Σ(σ_i² + d(c_i, c_j)²)

其中:
  k: 聚类数量
  σ_i: 簇i的样本标准差
  d(c_i, c_j): 簇i和簇j的中心距离
```

**取值范围**: [0, ∞)
- **接近0**: 簇内紧凑，簇间分离好
- **值越大**: 聚类效果越差

**业务意义**:
- 衡量簇的紧凑度和分离度
- 用于算法选择和参数优化

### 6. Calinski-Harabasz指数

```python
from sklearn.metrics import calinski_harabasz_score

ch_index = calinski_harabasz_score(feature_matrix, labels)
```

**计算原理**:
```
CH = (BG / (k-1)) / (WG / (n-k))

其中:
  BG: 簇间方差
  WG: 簇内方差
  k: 聚类数量
  n: 样本数量
```

**取值范围**: [0, ∞)
- **值越大**: 簇间分离度越高，聚类效果越好
- **结合DB指数**: 同时考虑分离度和紧凑度

**业务意义**:
- 评估聚类的分离质量
- 适合球形聚类算法的评价

---

## 工作流程指南

### 1. 完整处理流程

```python
# 步骤1: 数据加载
extractor = SDKFeatureExtractor()
extractor.load_data('data/raw/')

# 步骤2: 特征提取
features = extractor.extract_all_features()

# 步骤3: 特征融合
fused_features = extractor.fuse_features(features)

# 步骤4: 可选降维
if config['pca_components']:
    reduced_features = pca_transform(fused_features)
else:
    reduced_features = fused_features

# 步骤5: 聚类分析
clusterer = AlgorithmFactory.create_clusterer('hdbscan', params)
labels = clusterer.fit_predict(reduced_features)

# 步骤6: 结果评估
evaluator = MetricsEvaluator(sample_info)
metrics = evaluator.evaluate(labels, reduced_features)

# 步骤7: 结果导出
save_results(labels, metrics, sample_info)
```

### 2. 流程编排

```python
class Orchestrator:
    def run_pipeline(self, config):
        # 1. 特征工程阶段
        extractor = SDKFeatureExtractor(config['feature'])
        features = extractor.run()
        
        # 2. 降维阶段
        reducer = AlgorithmFactory.create_reducer(
            config['dimension_reduction']['method'],
            config['dimension_reduction']['target_dim']
        )
        reduced_features = reducer.fit_transform(features['fused'])
        
        # 3. 聚类阶段
        clusterer = AlgorithmFactory.create_clusterer(
            config['clustering']['algorithm'],
            config['clustering']['params']
        )
        labels = clusterer.fit_predict(reduced_features)
        
        # 4. 评估阶段
        evaluator = MetricsEvaluator(features['sample_info'])
        metrics = evaluator.evaluate(labels, reduced_features)
        
        # 5. 输出阶段
        outputs = {
            'labels': labels,
            'metrics': metrics,
            'features': reduced_features,
            'sample_info': features['sample_info']
        }
        
        return outputs
```

### 3. 配置示例

```python
config = {
    # 特征工程配置
    'feature': {
        'max_features': 2000,
        'd2v_vector_size': 128,
        'pca_components': None  # 不降维
    },
    
    # 降维配置
    'dimension_reduction': {
        'method': None,  # 'pca', 'umap', or None
        'target_dim': 100
    },
    
    # 聚类配置
    'clustering': {
        'algorithm': 'hdbscan',
        'params': {
            'min_cluster_size': 15,
            'min_samples': 5
        }
    }
}
```

---

## 参数调优指南

### 1. 特征工程参数调优

#### 1.1 TF-IDF参数

| 参数 | 调优策略 | 推荐范围 |
|------|----------|---------|
| **max_features** | 根据数据集大小调整 | 1000-3000 |
| **ngram_range** | 根据序列长度调整 | (1, 2), (2, 3) |
| **min_df** | 过滤低频词 | 2-5 |
| **max_df** | 过滤高频词 | 0.9-0.95 |

**调优建议**:
- 大数据集: 增加`max_features`
- 长序列: 增加`ngram_range`的上限
- 高频词: 降低`max_df`

#### 1.2 Doc2Vec参数

| 参数 | 调优策略 | 推荐范围 |
|------|----------|---------|
| **vector_size** | 根据特征复杂度调整 | 64-256 |
| **window** | 根据序列长度调整 | 3-10 |
| **epochs** | 根据数据集大小调整 | 10-50 |

**调优建议**:
- 复杂逻辑: 增加`vector_size`
- 长上下文: 增加`window`
- 大数据集: 增加`epochs`

#### 1.3 特征权重调优

```python
# 根据业务需求调整权重
config = {
    'opcode_weight': 0.35,      # 行为逻辑最重要
    'import_weight': 0.20,       # 增加依赖关系权重
    'coordinate_weight': 0.10,   # 降低命名权重
    # ... 其他权重
}

# 确保权重总和为1.0
total_weight = sum(config.values())
assert abs(total_weight - 1.0) < 0.01
```

### 2. 聚类算法参数调优

#### 2.1 HDBSCAN调优

```python
# 网格搜索最佳参数
param_grid = {
    'min_cluster_size': [10, 15, 20, 25],
    'min_samples': [3, 5, 7, 10]
}

best_score = -1
best_params = {}

for min_size in param_grid['min_cluster_size']:
    for min_samples in param_grid['min_samples']:
        clusterer = HDBSCANClusterer(
            min_cluster_size=min_size,
            min_samples=min_samples
        )
        labels = clusterer.fit_predict(features)
        
        # 计算评估指标
        score = compute_score(labels, features)
        
        if score > best_score:
            best_score = score
            best_params = {
                'min_cluster_size': min_size,
                'min_samples': min_samples
            }

print(f"最佳参数: {best_params}, 得分: {best_score}")
```

#### 2.2 DBSCAN调优

```python
# 网格搜索最佳参数
param_grid = {
    'eps': [0.3, 0.5, 0.7, 1.0],
    'min_samples': [5, 10, 15, 20]
}

best_score = -1
best_params = {}

for eps in param_grid['eps']:
    for min_samples in param_grid['min_samples']:
        clusterer = DBSCANClusterer(
            eps=eps,
            min_samples=min_samples
        )
        labels = clusterer.fit_predict(features)
        score = compute_score(labels, features)
        
        if score > best_score:
            best_score = score
            best_params = {
                'eps': eps,
                'min_samples': min_samples
            }
```

#### 2.3 K-Means调优

```python
# 肘部法确定最佳k值
scores = []
for k in range(5, 30):  # 测试5-30个聚类
    clusterer = KMeansClusterer(n_clusters=k)
    labels = clusterer.fit_predict(features)
    score = compute_score(labels, features)
    scores.append(score)

# 绘制肘部曲线
import matplotlib.pyplot as plt
plt.plot(range(5, 30), scores)
plt.xlabel('Number of clusters')
plt.ylabel('Score')
plt.title('Elbow Method')
plt.show()
```

### 3. 性能优化建议

#### 3.1 大数据集优化

```python
# 1. 分批处理
def batch_process(samples, batch_size=1000):
    results = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        result = process_batch(batch)
        results.extend(result)
    return results

# 2. 使用稀疏矩阵
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000, sparse=True)

# 3. 并行计算
from multiprocessing import Pool
with Pool(processes=4) as pool:
    results = pool.map(process_sample, samples)
```

#### 3.2 内存优化

```python
# 1. 及时释放内存
def process_large_dataset(samples):
    for i, sample in enumerate(samples):
        # 处理单个样本
        result = process_sample(sample)
        save_result(result)
        
        # 立即释放引用
        if i % 100 == 0:
            import gc
            gc.collect()

# 2. 使用生成器
def batch_generator(samples, batch_size=1000):
    for i in range(0, len(samples), batch_size):
        yield samples[i:i+batch_size]

for batch in batch_generator(large_samples):
    process_batch(batch)
```

---

## 总结

本技术文档详细介绍了SDK聚类引擎的特征处理和聚类流程：

1. **六维特征融合**: 结合多种特征类型，全方位描述SDK
2. **深度学习嵌入**: Doc2Vec捕获指令级语义
3. **业务感知设计**: 提取业务语义，增强可解释性
4. **多算法支持**: HDBSCAN/DBSCAN/K-Means/层次聚类
5. **业务导向评估**: 基于Ground Truth的验证体系

通过这套技术方案，可以为HarmonyOS应用SDK提供精准、高效、可解释的聚类分析能力。
