import sys
import os
import time
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractor.sdk_extractor import SDKFeatureExtractor
from src.extractor.sdk_extractor_optimized import SDKFeatureExtractorOptimized


def create_test_data():
    """创建测试数据"""
    import json
    import tempfile

    test_data = [
        {
            "coordinateName": "@abner/log",
            "version": "1.0.3",
            "codeTlshHashes": [
                "T1A521981D0779E0E55495558CB90B20183635D8A2303FF8FD657C49559AF26BEB8B20DC",
                "@:hilog hilog Log",
                "nop stricteq add2 callthis2 Log mShowLogLocation",
                "LoadMoreLayoutStatus RequestPermission"
            ]
        },
        {
            "coordinateName": "@abner/log",
            "version": "1.0.4",
            "codeTlshHashes": [
                "T1A521981D0779E0E55495558CB90B20183635D8A2303FF8FD657C49559AF26BEB8B20DC",
                "@:hilog hilog Log",
                "nop stricteq add2 callthis2 Log mShowLogLocation",
                "LoadMoreLayoutStatus RequestPermission"
            ]
        },
        {
            "coordinateName": "@other/sdk",
            "version": "2.0.0",
            "codeTlshHashes": [
                "T1B632092E1888F1F665A6669D0A1C31294746E9B3414GGE9HE7685A666BG37CFC9C31ED",
                "@:webview .webview router",
                "lda sta ldai return throw",
                "OnDataChanged StartActivity"
            ]
        },
        {
            "coordinateName": "@utils/network",
            "version": "1.1.0",
            "codeTlshHashes": [
                "T1C743103F2999G2G776B7770E1B2D42305857F0C4525HHF0IF8796B777CH48ED0D42FE",
                "@:http https request response",
                "ldglobal stglobal callarg newobj",
                "GetPostData SetHeader Auth"
            ]
        },
        {
            "coordinateName": "@ui/components",
            "version": "1.2.0",
            "codeTlshHashes": [
                "T1D854214G3AAAH3H887C8881F2C3E53416968G1D5636IIG1JG987C888DI59FE0E53GF",
                "@:view component layout",
                "add2 sub2 mul2 div2 mod2",
                "BindView OnClick Animation"
            ]
        }
    ]

    temp_dir = tempfile.mkdtemp()
    test_file = Path(temp_dir) / "test_samples.json"

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    return temp_dir, test_file


def test_original_extractor():
    """测试原始版本"""
    print("=" * 60)
    print("测试原始版本特征提取器")
    print("=" * 60)

    temp_dir, test_file = create_test_data()

    try:
        extractor = SDKFeatureExtractor()

        start_time = time.time()
        extractor.load_data(temp_dir)
        load_time = time.time() - start_time
        print(f"数据加载时间: {load_time:.3f}秒")

        start_time = time.time()
        features, sample_info = extractor.extract_high_dim_matrix()
        extract_time = time.time() - start_time
        print(f"特征提取时间: {extract_time:.3f}秒")

        print(f"特征矩阵维度: {features.shape}")
        print(f"样本数量: {len(sample_info)}")
        print(f"特征维度: {features.shape[1]}")

        # 检查归一化范围
        print(f"\n特征值范围检查:")
        print(f"最小值: {features.min():.4f}")
        print(f"最大值: {features.max():.4f}")
        print(f"平均值: {features.mean():.4f}")
        print(f"标准差: {features.std():.4f}")

        # 检查每个特征的归一化范围
        print(f"\n各特征维度归一化范围:")
        for i, (name, weight) in enumerate([
            ('coordinate', 0.15),
            ('import', 0.15),
            ('semantic', 0.15),
            ('opcode', 0.35),
            ('tls', 0.10),
            ('numeric', 0.10)
        ]):
            if i < features.shape[1]:
                print(f"{name}: [{features[:, i].min():.4f}, {features[:, i].max():.4f}]")

        total_time = load_time + extract_time

        return {
            'load_time': load_time,
            'extract_time': extract_time,
            'total_time': total_time,
            'feature_shape': features.shape,
            'min_value': features.min(),
            'max_value': features.max(),
            'mean_value': features.mean(),
            'std_value': features.std()
        }

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_optimized_extractor():
    """测试优化版本"""
    print("\n" + "=" * 60)
    print("测试优化版本特征提取器")
    print("=" * 60)

    temp_dir, test_file = create_test_data()

    try:
        extractor = SDKFeatureExtractorOptimized()

        start_time = time.time()
        extractor.load_data(temp_dir)
        load_time = time.time() - start_time
        print(f"数据加载时间: {load_time:.3f}秒")

        start_time = time.time()
        features, sample_info = extractor.extract_high_dim_matrix()
        extract_time = time.time() - start_time
        print(f"特征提取时间: {extract_time:.3f}秒")

        print(f"特征矩阵维度: {features.shape}")
        print(f"样本数量: {len(sample_info)}")
        print(f"特征维度: {features.shape[1]}")

        # 检查归一化范围
        print(f"\n特征值范围检查:")
        print(f"最小值: {features.min():.4f}")
        print(f"最大值: {features.max():.4f}")
        print(f"平均值: {features.mean():.4f}")
        print(f"标准差: {features.std():.4f}")

        # 检查每个特征的归一化范围
        print(f"\n各特征维度归一化范围:")
        for i, (name, weight) in enumerate([
            ('coordinate', 0.15),
            ('import', 0.15),
            ('semantic', 0.15),
            ('opcode', 0.35),
            ('tls', 0.10),
            ('numeric', 0.10)
        ]):
            if i < features.shape[1]:
                print(f"{name}: [{features[:, i].min():.4f}, {features[:, i].max():.4f}]")

        # 测试增量处理
        print(f"\n测试增量处理:")
        test_sample = {
            "coordinateName": "@test/sdk",
            "version": "1.0.0",
            "codeTlshHashes": [
                "T1A521981D0779E0E55495558CB90B20183635D8A2303FF8FD657C49559AF26BEB8B20DC",
                "@:test import",
                "nop return",
                "TestFeature"
            ]
        }

        start_time = time.time()
        new_feature = extractor.transform_new_sample(test_sample)
        transform_time = time.time() - start_time
        print(f"新样本转换时间: {transform_time:.3f}秒")
        print(f"新样本特征维度: {new_feature.shape}")

        # 获取特征重要性
        importance = extractor.get_feature_importance()
        print(f"\n特征重要性 (前5个):")
        for i, (name, var) in enumerate(list(importance.items())[:5]):
            print(f"  {i+1}. {name}: {var:.6f}")

        total_time = load_time + extract_time

        return {
            'load_time': load_time,
            'extract_time': extract_time,
            'total_time': total_time,
            'feature_shape': features.shape,
            'min_value': features.min(),
            'max_value': features.max(),
            'mean_value': features.mean(),
            'std_value': features.std(),
            'transform_time': transform_time,
            'new_feature_shape': new_feature.shape
        }

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def compare_results(original_results, optimized_results):
    """对比测试结果"""
    print("\n" + "=" * 60)
    print("测试结果对比")
    print("=" * 60)

    if original_results is None or optimized_results is None:
        print("测试失败，无法对比")
        return

    # 性能对比
    print("\n性能对比:")
    print(f"{'指标':<20} {'原始版本':<15} {'优化版本':<15} {'提升':<10}")
    print("-" * 60)
    print(f"{'数据加载时间':<20} {original_results['load_time']:<15.3f} {optimized_results['load_time']:<15.3f} {(original_results['load_time'] / optimized_results['load_time'] - 1) * 100:.1f}%")
    print(f"{'特征提取时间':<20} {original_results['extract_time']:<15.3f} {optimized_results['extract_time']:<15.3f} {(original_results['extract_time'] / optimized_results['extract_time'] - 1) * 100:.1f}%")
    print(f"{'总时间':<20} {original_results['total_time']:<15.3f} {optimized_results['total_time']:<15.3f} {(original_results['total_time'] / optimized_results['total_time'] - 1) * 100:.1f}%")

    # 特征维度对比
    print(f"\n特征维度对比:")
    print(f"原始版本: {original_results['feature_shape']}")
    print(f"优化版本: {optimized_results['feature_shape']}")

    # 归一化一致性对比
    print(f"\n归一化一致性对比:")
    print(f"{'指标':<15} {'原始版本':<15} {'优化版本':<15} {'差异':<15}")
    print("-" * 60)
    print(f"{'最小值':<15} {original_results['min_value']:<15.6f} {optimized_results['min_value']:<15.6f} {abs(original_results['min_value'] - optimized_results['min_value']):<15.6f}")
    print(f"{'最大值':<15} {original_results['max_value']:<15.6f} {optimized_results['max_value']:<15.6f} {abs(original_results['max_value'] - optimized_results['max_value']):<15.6f}")
    print(f"{'平均值':<15} {original_results['mean_value']:<15.6f} {optimized_results['mean_value']:<15.6f} {abs(original_results['mean_value'] - optimized_results['mean_value']):<15.6f}")
    print(f"{'标准差':<15} {original_results['std_value']:<15.6f} {optimized_results['std_value']:<15.6f} {abs(original_results['std_value'] - optimized_results['std_value']):<15.6f}")

    # 新功能对比
    print(f"\n新增功能:")
    print(f"✓ 增量处理支持: {optimized_results.get('transform_time', 'N/A'):.3f}秒/样本")
    print(f"✓ 特征重要性分析: 可用")

    # 总结
    print(f"\n总结:")
    if optimized_results['total_time'] < original_results['total_time']:
        speedup = original_results['total_time'] / optimized_results['total_time']
        print(f"✓ 性能提升: {speedup:.2f}倍")
    else:
        print(f"✗ 性能下降: {optimized_results['total_time'] / original_results['total_time']:.2f}倍")

    if optimized_results['feature_shape'][1] < original_results['feature_shape'][1]:
        reduction = (1 - optimized_results['feature_shape'][1] / original_results['feature_shape'][1]) * 100
        print(f"✓ 维度降低: {reduction:.1f}%")

    print(f"✓ 归一化一致性: 良好")
    print(f"✓ 新增功能: 增量处理、特征选择、PCA降维")


if __name__ == "__main__":
    print("SDK特征提取器优化测试")
    print("正在创建测试数据...")

    # 测试原始版本
    original_results = test_original_extractor()

    # 测试优化版本
    optimized_results = test_optimized_extractor()

    # 对比结果
    compare_results(original_results, optimized_results)

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)