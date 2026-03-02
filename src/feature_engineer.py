import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import hamming, cosine
import re
import warnings
import concurrent.futures
warnings.filterwarnings('ignore')


class SDKFeatureEngineer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.data = []
        self.feature_matrix = None
        self.scaler = None
        self.pca = None
        self.vocabulary = {'import_export': None, 'opcode': None}
        
    def _default_config(self) -> Dict:
        return {
            'max_features': 2000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95,
            'pca_components': None,
            'scaler': 'standard',
            'tlsh_weight': 0.3,
            'opcode_weight': 0.4,
            'import_weight': 0.3
        }
    
    def load_data(self, data_path: str) -> List[Dict]:
        if Path(data_path).suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.endswith('.jsonl'):
                    data = [json.loads(line.strip()) for line in f if line.strip()]
                else:
                    content = f.read()
                    if content.strip().startswith('['):
                        data = json.loads(content)
                    else:
                        data = [json.loads(content)]
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        self.data = self._validate_data(data)
        print(f"Loaded {len(self.data)} samples")
        return self.data
    
    def load_data_from_directory(self, directory_path: str) -> List[Dict]:
        """
        从目录中加载所有JSON文件（并行处理）
        """
        self.data = []
        directory = Path(directory_path)
        json_files = list(directory.glob('*.json'))
        
        print(f"Found {len(json_files)} JSON files in {directory_path}")
        
        def process_file(json_file):
            samples = []
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 尝试解析为单条JSON
                    try:
                        sample = json.loads(content)
                        if isinstance(sample, dict):
                            samples.append(sample)
                        elif isinstance(sample, list):
                            samples.extend(sample)
                    except json.JSONDecodeError:
                        # 尝试按行解析JSONL
                        lines = content.strip().split('\n')
                        for line in lines:
                            if line.strip():
                                try:
                                    sample = json.loads(line)
                                    if isinstance(sample, dict):
                                        samples.append(sample)
                                except:
                                    pass
            except Exception as e:
                print(f"Error processing file {json_file}: {e}")
            return samples
        
        # 并行处理文件
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(process_file, json_files)
            for samples in results:
                self.data.extend(samples)
        
        self.data = self._validate_data(self.data)
        print(f"Loaded {len(self.data)} samples from directory {directory_path}")
        return self.data
    
    def _validate_data(self, data: List[Dict]) -> List[Dict]:
        valid_data = []
        for i, item in enumerate(data):
            if 'coordinateName' not in item or 'codeTlshHashes' not in item:
                print(f"Warning: Sample {i} missing required fields, skipping")
                continue
            if not item['codeTlshHashes']:
                print(f"Warning: Sample {i} has empty codeTlshHashes, skipping")
                continue
            valid_data.append(item)
        return valid_data
    
    def _detect_sequence_type(self, sequence: str) -> str:
        if sequence.startswith('T1') and len(sequence) == 70:
            return 'tlsh'
        elif ':' in sequence and sequence.count(' ') < 5:
            return 'import_export'
        elif 'nop' in sequence or 'ld' in sequence or 'st' in sequence:
            return 'opcode'
        else:
            return 'unknown'
    
    def extract_features_per_sample(self, sample: Dict) -> Dict:
        hashes = sample['codeTlshHashes']
        features = {
            'coordinateName': sample['coordinateName'],
            'version': sample.get('version', 'unknown'),
            'import_export_sequences': [],
            'opcode_sequences': [],
            'tlsh_hashes': [],
            'total_sequences': len(hashes)
        }
        
        for seq in hashes:
            seq_type = self._detect_sequence_type(seq)
            if seq_type == 'import_export':
                features['import_export_sequences'].append(seq)
            elif seq_type == 'opcode':
                features['opcode_sequences'].append(seq)
            elif seq_type == 'tlsh':
                features['tlsh_hashes'].append(seq)
        
        return features
    
    def tokenize_sequence(self, sequence: str) -> List[str]:
        tokens = sequence.split()
        tokens = [t for t in tokens if t and not re.match(r'^[┌─|]+$', t)]
        return tokens
    
    def prepare_text_corpus(self, field_name: str) -> List[str]:
        corpus = []
        for sample in self.data:
            extracted = self.extract_features_per_sample(sample)
            sequences = extracted.get(field_name, [])
            if sequences:
                combined = ' '.join([' '.join(self.tokenize_sequence(seq)) for seq in sequences])
            else:
                combined = ''
            corpus.append(combined)
        return corpus
    
    def fit_tfidf_vectorizer(self, corpus: List[str], field_name: str):
        n_samples = len(corpus)
        min_df = min(self.config['min_df'], n_samples)
        max_df = max(self.config['max_df'], min_df / n_samples if n_samples > 0 else 0)
        
        vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            ngram_range=self.config['ngram_range'],
            min_df=min_df,
            max_df=max_df,
            token_pattern=r'(?u)\b\w+\b',
            lowercase=False
        )
        vectorizer.fit(corpus)
        self.vocabulary[field_name] = vectorizer
        return vectorizer
    
    def transform_tfidf(self, corpus: List[str], vectorizer: TfidfVectorizer) -> np.ndarray:
        features = vectorizer.transform(corpus).toarray()
        return features
    
    def compute_tls_hamming_distance(self, hash1: str, hash2: str) -> float:
        if len(hash1) != len(hash2):
            return 1.0
        distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        return distance / len(hash1)
    
    def extract_tls_features(self) -> np.ndarray:
        tls_features = []
        for sample in self.data:
            extracted = self.extract_features_per_sample(sample)
            tls_hashes = extracted['tlsh_hashes']
            
            if tls_hashes:
                tls_vector = np.zeros(70)
                for h in tls_hashes:
                    try:
                        hex_bytes = bytes.fromhex(h[2:])
                        for i, byte in enumerate(hex_bytes):
                            tls_vector[i * 2] = (byte >> 4) & 0xF
                            tls_vector[i * 2 + 1] = byte & 0xF
                    except:
                        continue
                tls_features.append(tls_vector)
            else:
                tls_features.append(np.zeros(70))
        
        return np.array(tls_features)
    
    def extract_all_features(self) -> Dict:
        print("Extracting import/export features...")
        import_corpus = self.prepare_text_corpus('import_export_sequences')
        self.fit_tfidf_vectorizer(import_corpus, 'import_export')
        import_features = self.transform_tfidf(import_corpus, self.vocabulary['import_export'])
        
        print("Extracting opcode features...")
        opcode_corpus = self.prepare_text_corpus('opcode_sequences')
        self.fit_tfidf_vectorizer(opcode_corpus, 'opcode')
        opcode_features = self.transform_tfidf(opcode_corpus, self.vocabulary['opcode'])
        
        print("Extracting TLSH features...")
        tls_features = self.extract_tls_features()
        
        features = {
            'import_export': import_features,
            'opcode': opcode_features,
            'tlsh': tls_features
        }
        
        return features
    
    def fuse_features(self, features: Dict) -> np.ndarray:
        import_feat = features['import_export']
        opcode_feat = features['opcode']
        tls_feat = features['tlsh']
        
        import_weight = self.config['import_weight']
        opcode_weight = self.config['opcode_weight']
        tls_weight = self.config['tlsh_weight']
        
        import_normalized = MinMaxScaler().fit_transform(import_feat) if import_feat.shape[1] > 0 else np.zeros((len(self.data), 1))
        opcode_normalized = MinMaxScaler().fit_transform(opcode_feat) if opcode_feat.shape[1] > 0 else np.zeros((len(self.data), 1))
        tls_normalized = MinMaxScaler().fit_transform(tls_feat) if tls_feat.shape[1] > 0 else np.zeros((len(self.data), 1))
        
        fused_features = np.hstack([
            import_normalized * import_weight,
            opcode_normalized * opcode_weight,
            tls_normalized * tls_weight
        ])
        
        return fused_features
    
    def apply_scaling(self, features: np.ndarray) -> np.ndarray:
        if self.config['scaler'] == 'standard':
            self.scaler = StandardScaler()
        elif self.config['scaler'] == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            return features
        
        return self.scaler.fit_transform(features)
    
    def apply_pca(self, features: np.ndarray) -> np.ndarray:
        if self.config['pca_components'] and self.config['pca_components'] < features.shape[1]:
            self.pca = PCA(n_components=self.config['pca_components'])
            features = self.pca.fit_transform(features)
            print(f"PCA reduced dimensions from {features.shape[1]} to {self.config['pca_components']}")
        return features
    
    def process(self, data_path: str) -> np.ndarray:
        print("Loading data...")
        path = Path(data_path)
        
        if path.is_dir():
            self.load_data_from_directory(data_path)
        else:
            self.load_data(data_path)
        
        print("\nExtracting features...")
        features_dict = self.extract_all_features()
        
        print(f"Import/Export features shape: {features_dict['import_export'].shape}")
        print(f"Opcode features shape: {features_dict['opcode'].shape}")
        print(f"TLSH features shape: {features_dict['tlsh'].shape}")
        
        print("\nFusing features...")
        self.feature_matrix = self.fuse_features(features_dict)
        print(f"Fused features shape: {self.feature_matrix.shape}")
        
        print("\nApplying scaling...")
        self.feature_matrix = self.apply_scaling(self.feature_matrix)
        
        print("\nApplying PCA (if configured)...")
        self.feature_matrix = self.apply_pca(self.feature_matrix)
        
        print(f"\nFinal feature matrix shape: {self.feature_matrix.shape}")
        
        return self.feature_matrix
    
    def get_sample_info(self, idx: int) -> Dict:
        if idx >= len(self.data):
            raise IndexError(f"Sample index {idx} out of range")
        
        sample = self.data[idx]
        extracted = self.extract_features_per_sample(sample)
        
        info = {
            'coordinateName': extracted['coordinateName'],
            'version': extracted['version'],
            'import_export_count': len(extracted['import_export_sequences']),
            'opcode_count': len(extracted['opcode_sequences']),
            'tlsh_count': len(extracted['tlsh_hashes']),
            'total_sequences': extracted['total_sequences']
        }
        
        return info
    
    def save_processed_data(self, output_path: str):
        output_data = {
            'feature_matrix': self.feature_matrix.tolist(),
            'sample_info': [self.get_sample_info(i) for i in range(len(self.data))],
            'config': self.config,
            'feature_shapes': {
                'import_export': self.vocabulary['import_export'].max_features if self.vocabulary['import_export'] else 0,
                'opcode': self.vocabulary['opcode'].max_features if self.vocabulary['opcode'] else 0,
                'tlsh': 70
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Processed data saved to {output_path}")
    
    def load_processed_data(self, input_path: str) -> np.ndarray:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.feature_matrix = np.array(data['feature_matrix'])
        self.config = data['config']
        
        print(f"Loaded processed data with shape: {self.feature_matrix.shape}")
        return self.feature_matrix
    
    def generate_report(self) -> Dict:
        if not self.data:
            return {}
        
        report = {
            'total_samples': len(self.data),
            'feature_matrix_shape': self.feature_matrix.shape if self.feature_matrix is not None else None,
            'sample_statistics': {
                'with_tls': 0,
                'with_import_export': 0,
                'with_opcode': 0,
                'avg_sequences': 0,
                'max_sequences': 0,
                'min_sequences': 0
            }
        }
        
        sequence_counts = []
        for sample in self.data:
            extracted = self.extract_features_per_sample(sample)
            sequence_counts.append(extracted['total_sequences'])
            
            if extracted['tlsh_hashes']:
                report['sample_statistics']['with_tls'] += 1
            if extracted['import_export_sequences']:
                report['sample_statistics']['with_import_export'] += 1
            if extracted['opcode_sequences']:
                report['sample_statistics']['with_opcode'] += 1
        
        if sequence_counts:
            report['sample_statistics']['avg_sequences'] = float(np.mean(sequence_counts))
            report['sample_statistics']['max_sequences'] = int(np.max(sequence_counts))
            report['sample_statistics']['min_sequences'] = int(np.min(sequence_counts))
        
        return report


def main():
    engineer = SDKFeatureEngineer()
    
    # 直接硬编码为包含7000多个JSON文件的目录路径
    data_path = '/Users/chengqihan/AICode/sdk_clustering/data/raw'
    output_path = '/Users/chengqihan/AICode/sdk_clustering/data/processed/processed_features.json'
    
    try:
        feature_matrix = engineer.process(data_path)
        
        engineer.save_processed_data(output_path)
        
        report = engineer.generate_report()
        print("\n" + "="*50)
        print("DATA PROCESSING REPORT")
        print("="*50)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        
    except FileNotFoundError:
        print(f"\n数据目录未找到: {data_path}")
        print(f"请将你的JSON数据文件放到: {data_path}")
        print(f"支持的格式:")
        print(f"  - 单个JSON文件")
        print(f"  - 目录中的多个JSON文件")
    except Exception as e:
        print(f"\n处理数据时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
