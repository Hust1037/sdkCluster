import json
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 引入深度学习文档嵌入模型
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class SDKFeatureExtractor:
    """
    SDK 特征工程流水线类 (混合架构终极版)
    核心改造：
    1. 引入 Doc2Vec 引擎处理操作码序列，捕获指令级上下文逻辑。
    2. 增加 business_semantic 维度，挽救有价值的业务常量标识符。
    3. 保留原有的 TLSH、Numeric 和 TF-IDF 基础架构，实现多维特征融合。
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.data = []
        self.vocabulary = {
            'coordinate': None,
            'import_export': None,
            'semantic': None
        }

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
            'd2v_vector_size': 128,  # 逻辑序列将被压缩成的核心维度
            'd2v_window': 5,  # 上下文滑动窗口大小
            'd2v_epochs': 20,  # 神经网络训练迭代次数

            # 混合特征权重分配 (总和 1.0)
            'coordinate_weight': 0.15,  # 名字引力
            'semantic_weight': 0.15,  # 业务语义常量 (如 LoadMoreLayoutStatus)
            'import_weight': 0.15,  # 外部依赖树
            'opcode_weight': 0.35,  # 指令深度逻辑 (赋予最高权重)
            'tlsh_weight': 0.10,  # 模糊哈希
            'numeric_weight': 0.10  # 数字序列特征
        }

    def load_data(self, directory_path: str):
        self.data = []
        directory = Path(directory_path)
        json_files = list(directory.glob('*.json')) + list(directory.glob('*.jsonl'))
        print(f"扫描目录: 发现 {len(json_files)} 个 JSON/JSONL 数据文件。")

        all_samples = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        samples = [parsed]
                    elif isinstance(parsed, list):
                        samples = parsed
                    else:
                        samples = []
                except json.JSONDecodeError:
                    lines = content.strip().split('\n')
                    samples = [json.loads(line) for line in lines if line.strip()]
                all_samples.extend(samples)
            except Exception as e:
                print(f"读取文件 {json_file} 失败: {e}")
                continue

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
            'throw', 'tryldglobal', 'nop', 'copyrestargs',
            # 新增：更多操作码关键词
            'newobjrange', 'defineclasswithbuffer', 'definemethod',
            'ldhole', 'stmodulevar', 'callthis0', 'callthis1', 'callthis2', 'callthis3',
            'createobjectwithbuffer', 'definefieldbyname', 'getInstance'
        }

        opcode_count = 0
        for token in tokens:
            token_lower = token.lower()
            if any(token_lower.startswith(keyword) for keyword in ARK_OPCODE_KEYWORDS):
                opcode_count += 1

        # 降低阈值：只要有操作码就判定为操作码序列
        # 因为操作码序列通常混合了操作码和变量名/属性名
        if opcode_count >= 1:
            return 'opcode'

        # 6. 兜底规则
        # 6.1 如果包含小写字母为主的token，判定为操作码
        if all(t.islower() and t.isalnum() and len(t) <= 15 for t in tokens[:5]):
            return 'opcode'

        # 6.2 属性访问模式（如 mVideoWidth, mLogOpen）
        # 这类序列通常是操作码上下文中的变量/属性名
        if len(tokens) == 1:
            token = tokens[0]
            # 检查是否是属性访问模式：m前缀 + 大写字母开头
            if re.match(r'^m[A-Z][a-zA-Z0-9_]*$', token):
                return 'opcode'
            # 检查是否是下划线开头的私有属性
            if re.match(r'^_[a-zA-Z_][a-zA-Z0-9_]*$', token):
                return 'opcode'

        return 'unknown'

    def extract_features_per_sample(self, sample: Dict) -> Dict:
        hashes = sample['codeTlshHashes']
        features = {
            'coordinateName': sample['coordinateName'],
            'version': sample.get('version', ''),
            'import_export_sequences': [],
            'semantic_sequences': [],  # 新增
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
        """专门为 TF-IDF 构建纯文本语料库"""
        corpus = []
        for sample in self.data:
            extracted = self.extract_features_per_sample(sample)
            sequences = extracted.get(field_name, [])
            combined = ' '.join([' '.join(self.tokenize_sequence(seq)) for seq in sequences]) if sequences else ''
            corpus.append(combined)
        return corpus

    def _train_doc2vec_for_opcodes(self) -> np.ndarray:
        """专门利用 Doc2Vec 为操作码序列生成高维稠密逻辑向量"""
        print("启动 Doc2Vec 引擎，正在对底层汇编指令执行流进行上下文建模...")

        tagged_docs = []
        for i, sample in enumerate(self.data):
            extracted = self.extract_features_per_sample(sample)
            opcodes = extracted.get('opcode_sequences', [])

            # 将 SDK 的所有指令序列打平为一个文档
            tokens = []
            for seq in opcodes:
                tokens.extend(self.tokenize_sequence(seq))

            # 占位符防止空崩溃
            if not tokens:
                tokens = ['__empty_opcode__']

            tagged_docs.append(TaggedDocument(words=tokens, tags=[str(i)]))

        # 初始化并训练 PV-DM 模型
        model = Doc2Vec(
            vector_size=self.config['d2v_vector_size'],
            window=self.config['d2v_window'],
            min_count=2,
            workers=4,
            epochs=self.config['d2v_epochs'],
            dm=1
        )

        model.build_vocab(tagged_docs)
        model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

        # 按样本索引提取训练好的特征向量
        opcode_matrix = np.array([model.dv[str(i)] for i in range(len(self.data))])
        print(f"Doc2Vec 逻辑向量提取完毕，特征矩阵维度: {opcode_matrix.shape}")
        return opcode_matrix

    def extract_tls_features(self) -> np.ndarray:
        # 保持原样
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
        # 保持原样
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
        """
        print("开始提取 CoordinateName 模糊字元特征...")
        coord_corpus = [sample['coordinateName'] for sample in self.data]
        v_coord = TfidfVectorizer(analyzer='char_wb', ngram_range=self.config['coordinate_ngram'], lowercase=True,
                                  max_features=1000)
        coord_features = v_coord.fit_transform(coord_corpus).toarray()

        print("开始提取 导入导出依赖树(CDG)特征...")
        import_corpus = self.prepare_text_corpus('import_export_sequences')
        v_imp = TfidfVectorizer(max_features=self.config['max_features'], ngram_range=self.config['ngram_range'],
                                min_df=2, max_df=0.95)
        import_features = v_imp.fit_transform(import_corpus).toarray() if any(import_corpus) else np.zeros(
            (len(self.data), 1))

        print("开始提取 业务语义常量(Semantic)特征...")
        semantic_corpus = self.prepare_text_corpus('semantic_sequences')
        # 业务常量通常看独立单词即可，故使用 (1, 1)
        v_sem = TfidfVectorizer(max_features=1000, ngram_range=(1, 1), min_df=2, max_df=0.95)
        semantic_features = v_sem.fit_transform(semantic_corpus).toarray() if any(semantic_corpus) else np.zeros(
            (len(self.data), 1))

        # Doc2Vec 处理操作码序列
        opcode_features = self._train_doc2vec_for_opcodes()

        print("开始处理 (TLSH)与数字序列特征...")
        tls_features = self.extract_tls_features()
        numeric_features = self.extract_numeric_features()

        print("执行特征加权融合与数值归一化...")
        w = self.config

        def scale(matrix):
            return MinMaxScaler().fit_transform(matrix) if matrix.shape[1] > 0 else np.zeros((len(self.data), 1))

        # 横向堆叠所有维度的特征矩阵
        fused_matrix = np.hstack([
            scale(coord_features) * w['coordinate_weight'],
            scale(import_features) * w['import_weight'],
            scale(semantic_features) * w['semantic_weight'],
            scale(opcode_features) * w['opcode_weight'],
            scale(tls_features) * w['tlsh_weight'],
            scale(numeric_features) * w['numeric_weight']
        ])

        sample_info = []
        for i, sample in enumerate(self.data):
            extracted = self.extract_features_per_sample(sample)
            sample_info.append({
                'coordinateName': extracted['coordinateName'],
                'version': extracted['version']
            })

        print(f"混合高维特征矩阵构建完毕，总维度: {fused_matrix.shape}")
        return fused_matrix, sample_info

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