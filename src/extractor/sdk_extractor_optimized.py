import json
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class SDKFeatureExtractorOptimized:
    """
    SDK 特征工程流水线类 (优化版)
    核心改进：
    1. 修复Scaler一致性问题
    2. 增加并行处理支持
    3. 添加特征选择和降维
    4. 优化内存使用
    5. 添加增量处理支持
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.data = []
        self.vectorizers = {}
        self.scalers = {}
        self.pca_model = None
        self.feature_selector = None

    def _default_config(self) -> Dict:
        return {
            # TF-IDF 基础参数
            'max_features': 2000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95,
            'coordinate_ngram': (3, 5),
            'max_numeric_len': 128,

            # Doc2Vec 核心参数
            'd2v_vector_size': 128,
            'd2v_window': 5,
            'd2v_epochs': 20,

            # 混合特征权重分配 (总和 1.0)
            'coordinate_weight': 0.15,
            'semantic_weight': 0.15,
            'import_weight': 0.15,
            'opcode_weight': 0.35,
            'tlsh_weight': 0.10,
            'numeric_weight': 0.10,

            # 新增优化参数
            'enable_feature_selection': True,
            'variance_threshold': 0.01,
            'enable_pca': True,
            'pca_components': 100,
            'use_sparse': True,
            'n_jobs': 4
        }

    def load_data(self, directory_path: str):
        """优化版本：添加并行文件读取"""
        from concurrent.futures import ThreadPoolExecutor

        self.data = []
        directory = Path(directory_path)
        json_files = list(directory.glob('*.json')) + list(directory.glob('*.jsonl'))
        print(f"扫描目录: 发现 {len(json_files)} 个 JSON/JSONL 数据文件。")

        def load_file(json_file):
            samples = []
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        samples = [parsed]
                    elif isinstance(parsed, list):
                        samples = parsed
                except json.JSONDecodeError:
                    lines = content.strip().split('\n')
                    samples = [json.loads(line) for line in lines if line.strip()]
            except Exception as e:
                print(f"读取文件 {json_file} 失败: {e}")
            return samples

        # 并行读取文件
        with ThreadPoolExecutor(max_workers=self.config['n_jobs']) as executor:
            results = executor.map(load_file, json_files)
            all_samples = []
            for samples in results:
                all_samples.extend(samples)

        # 验证数据
        valid_data = []
        for item in all_samples:
            if not isinstance(item, dict): continue
            if 'features' in item and 'codeTlshHashes' not in item: continue
            if 'coordinateName' not in item or 'codeTlshHashes' not in item: continue

            hashes = item['codeTlshHashes']
            if not isinstance(hashes, list) or len(hashes) == 0: continue
            if not all(isinstance(seq, str) for seq in hashes): continue

            valid_data.append(item)

        self.data = valid_data
        print(f"数据加载与清洗完成，有效样本数: {len(self.data)}")

    def _detect_sequence_type(self, sequence: str) -> str:
        s = sequence.strip()
        if not s: return 'unknown'

        # 1. TLSH检测（支持70-72字符长度）
        if s.startswith('T1') and 70 <= len(s) <= 72 and all(c in '0123456789ABCDEF' for c in s[2:]):
            return 'tlsh'

        tokens = s.split()
        if not tokens: return 'unknown'

        # 2. 业务语义检测（优先级最高）
        business_semantic_indicators = 0

        # 2.1 函数调用模式 _get[A-Z], _set[A-Z]
        if re.search(r'_[gs]et[A-Z][a-zA-Z]*', s):
            business_semantic_indicators += 2

        # 2.2 事件监听模式 mOn[A-Z]
        if re.search(r'mOn[A-Z][a-zA-Z]+', s):
            business_semantic_indicators += 2

        # 2.3 业务关键词
        business_keywords = ['PROP_', 'FFP_', 'Listener', 'Handler']
        if any(keyword in s for keyword in business_keywords):
            business_semantic_indicators += 2

        # 2.4 特殊业务函数
        business_functions = ['definefieldbyname', 'createobjectwithbuffer', 'getInstance']
        if any(func in s for func in business_functions):
            business_semantic_indicators += 2

        # 2.5 包含版本号的类型引用
        if re.search(r'[A-Z][a-zA-Z]+&[0-9]+\.[0-9]+', s):
            business_semantic_indicators += 1

        # 如果业务语义指示器>=2，判定为业务语义
        if business_semantic_indicators >= 2:
            return 'business_semantic'

        # 3. 导入导出检测（支持多种格式）
        import_export_indicators = 0

        # 3.1 标准@:格式
        if '@:' in s:
            import_export_indicators += 3

        # 3.2 归一化声明 @normalized
        if '@normalized' in s:
            import_export_indicators += 2

        # 3.3 路径引用 @/path/to/file
        if re.search(r'@/[a-zA-Z0-9_./-]+', s):
            import_export_indicators += 2

        # 3.4 库文件引用 libname.so
        if re.search(r'[a-zA-Z0-9_]+\.so', s):
            import_export_indicators += 2

        # 3.5 其他导入导出关键词
        import_keywords = ['import', 'export', 'require', 'declare', '.webview', 'router']
        if any(keyword in s.lower() for keyword in import_keywords):
            import_export_indicators += 2

        # 3.6 特殊分隔符 &&& (连接多个导入导出)
        if '&&&' in s:
            import_export_indicators += 2

        # 3.7 包含版本号 &版本格式
        if re.search(r'&[0-9]+\.[0-9]+', s):
            import_export_indicators += 1

        # 如果导入导出指示器>=2，判定为导入导出
        if import_export_indicators >= 2:
            return 'import_export'

        # 4. 数字序列检测
        if all(token.isdigit() for token in tokens):
            return 'numeric'

        # 5. 操作码检测
        ARK_OPCODE_KEYWORDS = {
            'lda', 'sta', 'ldai', 'ldglobal', 'stglobal', 'ldobj', 'stobj',
            'ldlexvar', 'stlexvar', 'newlexenv', 'poplexenv',
            'callthis', 'returnundefined', 'istrue', 'ldfalse', 'isfalse',
            'add2', 'sub2', 'mul2', 'div2', 'mod2', 'shl2', 'shr2', 'ashr2',
            'and2', 'or2', 'xor2', 'stricteq', 'strictnoteq',
            'throw', 'tryldglobal', 'nop', 'copyrestargs'
        }

        opcode_count = 0
        for token in tokens:
            if any(token.lower().startswith(keyword) for keyword in ARK_OPCODE_KEYWORDS):
                opcode_count += 1
            if opcode_count >= 3:  # 包含3个以上操作码
                return 'opcode'

        # 6. 兜底：如果包含小写字母为主的token，判定为操作码
        if all(t.islower() and t.isalnum() and len(t) <= 15 for t in tokens[:5]):
            return 'opcode'

        return 'unknown'

    def extract_features_per_sample(self, sample: Dict) -> Dict:
        hashes = sample['codeTlshHashes']
        features = {
            'coordinateName': sample['coordinateName'],
            'version': sample.get('version', ''),
            'import_export_sequences': [],
            'semantic_sequences': [],
            'opcode_sequences': [],
            'tlsh_hashes': [],
            'numeric_sequences': [],
            'unknown_sequences': []  # 新增：记录无法识别的序列
        }
        for seq in hashes:
            seq_type = self._detect_sequence_type(seq)
            if seq_type == 'import_export':
                features['import_export_sequences'].append(seq)
            elif seq_type == 'business_semantic':
                features['semantic_sequences'].append(seq)
            elif seq_type == 'opcode':
                features['opcode_sequences'].append(seq)
            elif seq_type == 'tlsh':
                features['tlsh_hashes'].append(seq)
            elif seq_type == 'numeric':
                features['numeric_sequences'].append(seq)
            else:
                features['unknown_sequences'].append(seq)
        return features

    def tokenize_sequence(self, sequence: str) -> List[str]:
        tokens = sequence.split()
        tokens = [t for t in tokens if t and not re.match(r'^[┌─|`]+$|^\u0000$', t)]
        return tokens

    def prepare_text_corpus(self, field_name: str) -> List[str]:
        corpus = []
        for sample in self.data:
            extracted = self.extract_features_per_sample(sample)
            sequences = extracted.get(field_name, [])
            combined = ' '.join([' '.join(self.tokenize_sequence(seq)) for seq in sequences]) if sequences else ''
            corpus.append(combined)
        return corpus

    def _train_doc2vec_for_opcodes(self) -> np.ndarray:
        """优化版本：减少训练epochs，提升训练速度"""
        print("启动 Doc2Vec 引擎，正在对底层汇编指令执行流进行上下文建模...")

        tagged_docs = []
        for i, sample in enumerate(self.data):
            extracted = self.extract_features_per_sample(sample)
            opcodes = extracted.get('opcode_sequences', [])

            tokens = []
            for seq in opcodes:
                tokens.extend(self.tokenize_sequence(seq))

            if not tokens:
                tokens = ['__empty_opcode__']

            tagged_docs.append(TaggedDocument(words=tokens, tags=[str(i)]))

        # 优化：减少epochs，提升训练速度
        model = Doc2Vec(
            vector_size=self.config['d2v_vector_size'],
            window=self.config['d2v_window'],
            min_count=2,
            workers=4,
            epochs=min(self.config['d2v_epochs'], 10),
            dm=1
        )

        model.build_vocab(tagged_docs)
        model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

        opcode_matrix = np.array([model.dv[str(i)] for i in range(len(self.data))])
        print(f"Doc2Vec 逻辑向量提取完毕，特征矩阵维度: {opcode_matrix.shape}")
        return opcode_matrix

    def extract_tls_features(self) -> np.ndarray:
        tls_features = []
        for sample in self.data:
            extracted = self.extract_features_per_sample(sample)
            tls_hashes = extracted['tlsh_hashes']
            if tls_hashes:
                sample_tls_vectors = []
                for h in tls_hashes:
                    try:
                        hex_part = h[2:]
                        if len(hex_part) == 68:
                            hex_bytes = bytes.fromhex(hex_part)
                            vec = np.zeros(70)
                            for i, byte in enumerate(hex_bytes):
                                if i * 2 + 1 < 70:
                                    vec[i * 2] = (byte >> 4) & 0xF
                                    vec[i * 2 + 1] = byte & 0xF
                            sample_tls_vectors.append(vec)
                    except Exception:
                        continue
                if sample_tls_vectors:
                    final_tls_vector = np.mean(sample_tls_vectors, axis=0)
                    tls_features.append(final_tls_vector)
                else:
                    tls_features.append(np.zeros(70))
            else:
                tls_features.append(np.zeros(70))
        return np.array(tls_features)

    def extract_numeric_features(self) -> np.ndarray:
        max_len = self.config['max_numeric_len']
        features = []
        for sample in self.data:
            extracted = self.extract_features_per_sample(sample)
            seqs = extracted['numeric_sequences']
            if seqs:
                all_nums = []
                for s in seqs:
                    nums = [int(x) for x in s.split() if x.isdigit()]
                    all_nums.extend(nums)
                vec = all_nums[:max_len] if len(all_nums) > max_len else all_nums + [0] * (max_len - len(all_nums))
            else:
                vec = [0] * max_len
            features.append(vec)
        return np.array(features, dtype=np.float32)

    def extract_high_dim_matrix(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        核心输出方法：依次执行异构特征抽取，并使用归一化加权融合。
        优化版本：修复Scaler一致性问题，添加特征选择和降维。
        """
        print("开始提取 CoordinateName 模糊字元特征...")
        coord_corpus = [sample['coordinateName'] for sample in self.data]
        v_coord = TfidfVectorizer(analyzer='char_wb', ngram_range=self.config['coordinate_ngram'], lowercase=True,
                                  max_features=1000, sparse=self.config['use_sparse'])
        coord_features = v_coord.fit_transform(coord_corpus)
        if not self.config['use_sparse']:
            coord_features = coord_features.toarray()
        self.vectorizers['coordinate'] = v_coord

        print("开始提取 导入导出依赖树(CDG)特征...")
        import_corpus = self.prepare_text_corpus('import_export_sequences')
        v_imp = TfidfVectorizer(max_features=self.config['max_features'], ngram_range=self.config['ngram_range'],
                                min_df=2, max_df=0.95, sparse=self.config['use_sparse'])
        import_features = v_imp.fit_transform(import_corpus) if any(import_corpus) else np.zeros(
            (len(self.data), 1))
        if not self.config['use_sparse'] and hasattr(import_features, 'toarray'):
            import_features = import_features.toarray()
        self.vectorizers['import_export'] = v_imp

        print("开始提取 业务语义常量(Semantic)特征...")
        semantic_corpus = self.prepare_text_corpus('semantic_sequences')
        v_sem = TfidfVectorizer(max_features=1000, ngram_range=(1, 1), min_df=2, max_df=0.95,
                                sparse=self.config['use_sparse'])
        semantic_features = v_sem.fit_transform(semantic_corpus) if any(semantic_corpus) else np.zeros(
            (len(self.data), 1))
        if not self.config['use_sparse'] and hasattr(semantic_features, 'toarray'):
            semantic_features = semantic_features.toarray()
        self.vectorizers['semantic'] = v_sem

        # Doc2Vec 处理操作码序列
        opcode_features = self._train_doc2vec_for_opcodes()

        print("开始处理 (TLSH)与数字序列特征...")
        tls_features = self.extract_tls_features()
        numeric_features = self.extract_numeric_features()

        print("执行特征加权融合与数值归一化...")
        w = self.config

        # 关键优化：一次性创建所有scaler，确保一致性
        print("创建归一化器...")
        scaler_coord = MinMaxScaler()
        scaler_import = MinMaxScaler()
        scaler_semantic = MinMaxScaler()
        scaler_opcode = MinMaxScaler()
        scaler_tls = MinMaxScaler()
        scaler_numeric = StandardScaler()

        # 统一归一化处理
        print("执行归一化处理...")
        coord_normalized = scaler_coord.fit_transform(coord_features) if coord_features.shape[1] > 0 else np.zeros((len(self.data), 1))
        import_normalized = scaler_import.fit_transform(import_features) if import_features.shape[1] > 0 else np.zeros((len(self.data), 1))
        semantic_normalized = scaler_semantic.fit_transform(semantic_features) if semantic_features.shape[1] > 0 else np.zeros((len(self.data), 1))
        opcode_normalized = scaler_opcode.fit_transform(opcode_features) if opcode_features.shape[1] > 0 else np.zeros((len(self.data), 1))
        tls_normalized = scaler_tls.fit_transform(tls_features) if tls_features.shape[1] > 0 else np.zeros((len(self.data), 1))
        numeric_normalized = scaler_numeric.fit_transform(numeric_features) if numeric_features.shape[1] > 0 else np.zeros((len(self.data), 1))

        # 保存scaler用于后续增量处理
        self.scalers['coordinate'] = scaler_coord
        self.scalers['import_export'] = scaler_import
        self.scalers['semantic'] = scaler_semantic
        self.scalers['opcode'] = scaler_opcode
        self.scalers['tls'] = scaler_tls
        self.scalers['numeric'] = scaler_numeric

        # 横向堆叠所有维度的特征矩阵
        print("执行特征融合...")
        fused_matrix = np.hstack([
            coord_normalized * w['coordinate_weight'],
            import_normalized * w['import_weight'],
            semantic_normalized * w['semantic_weight'],
            opcode_normalized * w['opcode_weight'],
            tls_normalized * w['tlsh_weight'],
            numeric_normalized * w['numeric_weight']
        ])

        # 特征选择
        if self.config['enable_feature_selection']:
            print(f"执行特征选择 (方差阈值: {self.config['variance_threshold']})...")
            selector = VarianceThreshold(threshold=self.config['variance_threshold'])
            fused_matrix = selector.fit_transform(fused_matrix)
            self.feature_selector = selector
            print(f"特征选择完成: 保留 {fused_matrix.shape[1]} 维特征")

        # PCA降维
        if self.config['enable_pca'] and self.config['pca_components'] > 0:
            print(f"执行PCA降维 (目标维度: {self.config['pca_components']})...")
            pca = PCA(n_components=self.config['pca_components'], random_state=42)
            fused_matrix = pca.fit_transform(fused_matrix)
            self.pca_model = pca
            variance_ratio = pca.explained_variance_ratio_.sum()
            print(f"PCA降维完成: 保留 {variance_ratio:.2%} 方差, 最终维度: {fused_matrix.shape[1]}")

        sample_info = []
        for i, sample in enumerate(self.data):
            extracted = self.extract_features_per_sample(sample)
            sample_info.append({
                'coordinateName': extracted['coordinateName'],
                'version': extracted['version']
            })

        print(f"混合高维特征矩阵构建完毕，总维度: {fused_matrix.shape}")
        return fused_matrix, sample_info

    def transform_new_sample(self, sample: Dict) -> np.ndarray:
        """
        增量处理：将新样本转换为特征向量
        复用已训练的vectorizer、scaler、pca等
        """
        # 验证新样本
        if 'coordinateName' not in sample or 'codeTlshHashes' not in sample:
            raise ValueError("样本缺少必需字段")

        # 提取特征
        extracted = self.extract_features_per_sample({'coordinateName': sample['coordinateName'],
                                                     'codeTlshHashes': sample['codeTlshHashes']})

        # Transform各特征
        coord_features = self.vectorizers['coordinate'].transform([extracted['coordinateName']])
        if hasattr(coord_features, 'toarray'):
            coord_features = coord_features.toarray()

        import_corpus = ' '.join([' '.join(self.tokenize_sequence(seq))
                                 for seq in extracted['import_export_sequences']])
        import_features = self.vectorizers['import_export'].transform([import_corpus])
        if hasattr(import_features, 'toarray'):
            import_features = import_features.toarray()

        semantic_corpus = ' '.join([' '.join(self.tokenize_sequence(seq))
                                   for seq in extracted['semantic_sequences']])
        semantic_features = self.vectorizers['semantic'].transform([semantic_corpus])
        if hasattr(semantic_features, 'toarray'):
            semantic_features = semantic_features.toarray()

        # Opcode需要特殊处理
        tokens = []
        for seq in extracted['opcode_sequences']:
            tokens.extend(self.tokenize_sequence(seq))
        if not tokens:
            tokens = ['__empty_opcode__']
        opcode_features = self.scalers['opcode'].transform(np.array([np.zeros(self.config['d2v_vector_size'])]))

        # TLSH和Numeric特征
        if extracted['tlsh_hashes']:
            tls_vectors = []
            for h in extracted['tls_hashes']:
                hex_part = h[2:]
                if len(hex_part) == 68:
                    hex_bytes = bytes.fromhex(hex_part)
                    vec = np.zeros(70)
                    for i, byte in enumerate(hex_bytes):
                        if i * 2 + 1 < 70:
                            vec[i * 2] = (byte >> 4) & 0xF
                            vec[i * 2 + 1] = byte & 0xF
                    tls_vectors.append(vec)
            tls_features = np.mean(tls_vectors, axis=0).reshape(1, -1)
        else:
            tls_features = np.zeros((1, 70))

        max_len = self.config['max_numeric_len']
        if extracted['numeric_sequences']:
            all_nums = []
            for s in extracted['numeric_sequences']:
                nums = [int(x) for x in s.split() if x.isdigit()]
                all_nums.extend(nums)
            vec = all_nums[:max_len] if len(all_nums) > max_len else all_nums + [0] * (max_len - len(all_nums))
        else:
            vec = [0] * max_len
        numeric_features = np.array(vec, dtype=np.float32).reshape(1, -1)

        # 归一化
        w = self.config
        coord_normalized = self.scalers['coordinate'].transform(coord_features)
        import_normalized = self.scalers['import_export'].transform(import_features)
        semantic_normalized = self.scalers['semantic'].transform(semantic_features)
        tls_normalized = self.scalers['tls'].transform(tls_features)
        numeric_normalized = self.scalers['numeric'].transform(numeric_features)

        # 融合
        fused = np.hstack([
            coord_normalized * w['coordinate_weight'],
            import_normalized * w['import_weight'],
            semantic_normalized * w['semantic_weight'],
            opcode_features * w['opcode_weight'],
            tls_normalized * w['tlsh_weight'],
            numeric_normalized * w['numeric_weight']
        ])

        # 特征选择
        if self.feature_selector:
            fused = self.feature_selector.transform(fused)

        # PCA
        if self.pca_model:
            fused = self.pca_model.transform(fused)

        return fused

    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性（基于方差）
        """
        if not hasattr(self, 'feature_selector') or self.feature_selector is None:
            return {}

        variances = self.feature_selector.variances_
        feature_names = []

        # 按权重分配特征名称
        w = self.config
        n_coord = int(self.vectorizers['coordinate'].max_features * w['coordinate_weight'])
        n_import = int(self.vectorizers['import_export'].max_features * w['import_weight'])
        n_semantic = int(self.vectorizers['semantic'].max_features * w['semantic_weight'])
        n_opcode = self.config['d2v_vector_size']
        n_tls = 70
        n_numeric = self.config['max_numeric_len']

        feature_names.extend([f'coord_{i}' for i in range(n_coord)])
        feature_names.extend([f'import_{i}' for i in range(n_import)])
        feature_names.extend([f'semantic_{i}' for i in range(n_semantic)])
        feature_names.extend([f'opcode_{i}' for i in range(n_opcode)])
        feature_names.extend([f'tls_{i}' for i in range(n_tls)])
        feature_names.extend([f'numeric_{i}' for i in range(n_numeric)])

        importance = {name: var for name, var in zip(feature_names, variances)}
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save_unknown_sequences(self, output_path: str = 'unknown_sequences.json'):
        """
        保存无法识别的unknown序列到JSON文件
        用于后续分析和优化序列检测逻辑
        """
        unknown_data = []

        for sample in self.data:
            extracted = self.extract_features_per_sample(sample)
            unknown_seqs = extracted.get('unknown_sequences', [])

            if unknown_seqs:
                unknown_data.append({
                    'coordinateName': extracted['coordinateName'],
                    'version': extracted['version'],
                    'unknown_count': len(unknown_seqs),
                    'unknown_sequences': unknown_seqs,
                    'total_sequences': len(sample['codeTlshHashes'])
                })

        # 统计信息
        total_unknown = sum(item['unknown_count'] for item in unknown_data)
        total_samples = len(unknown_data)

        output = {
            'summary': {
                'total_samples_with_unknown': total_samples,
                'total_unknown_sequences': total_unknown,
                'unknown_rate': f"{total_unknown / total_samples * 100:.1f}%" if total_samples > 0 else "0%"
            },
            'unknown_data': unknown_data
        }

        # 保存到文件
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"已保存unknown序列到文件: {output_path}")
            print(f"  - 总样本数: {total_samples}")
            print(f"  - Unknown序列总数: {total_unknown}")
            print(f"  - Unknown比例: {output['summary']['unknown_rate']}")
        except Exception as e:
            print(f"保存unknown序列失败: {e}")