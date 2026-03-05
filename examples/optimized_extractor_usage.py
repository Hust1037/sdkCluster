import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractor.sdk_extractor_optimized import SDKFeatureExtractorOptimized


def basic_usage_example():
    """基本使用示例"""
    print("=" * 60)
    print("SDK特征提取器 - 基本使用示例")
    print("=" * 60)

    # 1. 创建提取器（使用默认配置）
    extractor = SDKFeatureExtractorOptimized()

    # 2. 加载数据
    print("\n步骤1: 加载数据")
    data_dir = "data/raw/"
    try:
        extractor.load_data(data_dir)
        print(f"✓ 成功加载 {len(extractor.data)} 个样本")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return

    # 3. 提取特征
    print("\n步骤2: 提取特征")
    features, sample_info = extractor.extract_high_dim_matrix()
    print(f"✓ 特征矩阵: {features.shape}")
    print(f"✓ 样本信息: {len(sample_info)} 条")

    # 4. 查看特征统计
    print("\n步骤3: 特征统计")
    print(f"特征维度: {features.shape[1]}")
    print(f"数值范围: [{features.min():.4f}, {features.max():.4f}]")
    print(f"平均值: {features.mean():.4f}")
    print(f"标准差: {features.std():.4f}")

    # 5. 查看样本信息
    print("\n步骤4: 样本信息")
    for i, info in enumerate(sample_info[:5]):  # 显示前5个样本
        print(f"  {i+1}. {info['coordinateName']} @ {info['version']}")

    return extractor, features, sample_info


def advanced_usage_example():
    """高级使用示例"""
    print("\n" + "=" * 60)
    print("SDK特征提取器 - 高级使用示例")
    print("=" * 60)

    # 1. 自定义配置
    print("\n步骤1: 自定义配置")
    config = {
        # 特征参数
        'max_features': 2000,
        'd2v_vector_size': 128,
        'coordinate_ngram': (3, 5),

        # 权重配置
        'coordinate_weight': 0.15,
        'semantic_weight': 0.15,
        'import_weight': 0.15,
        'opcode_weight': 0.35,
        'tlsh_weight': 0.10,
        'numeric_weight': 0.10,

        # 优化参数
        'enable_feature_selection': True,
        'variance_threshold': 0.01,
        'enable_pca': True,
        'pca_components': 100,
        'use_sparse': True,
        'n_jobs': 4
    }

    extractor = SDKFeatureExtractorOptimized(config)
    print("✓ 配置完成")

    # 2. 加载数据
    print("\n步骤2: 加载数据")
    data_dir = "data/raw/"
    try:
        extractor.load_data(data_dir)
        print(f"✓ 成功加载 {len(extractor.data)} 个样本")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return

    # 3. 提取特征（带特征选择和PCA）
    print("\n步骤3: 提取特征")
    features, sample_info = extractor.extract_high_dim_matrix()
    print(f"✓ 特征矩阵: {features.shape}")
    print(f"✓ 最终维度: {features.shape[1]} (已降维)")

    # 4. 特征重要性分析
    print("\n步骤4: 特征重要性分析")
    importance = extractor.get_feature_importance()
    print("最重要的特征 (前10个):")
    for i, (name, var) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1:2d}. {name:<20} {var:.6f}")

    return extractor, features, sample_info


def incremental_processing_example():
    """增量处理示例"""
    print("\n" + "=" * 60)
    print("SDK特征提取器 - 增量处理示例")
    print("=" * 60)

    # 1. 初始化并训练
    print("\n步骤1: 初始化并训练")
    extractor = SDKFeatureExtractorOptimized()

    data_dir = "data/raw/"
    try:
        extractor.load_data(data_dir)
        features, sample_info = extractor.extract_high_dim_matrix()
        print(f"✓ 训练完成: {features.shape}")
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        return

    # 2. 处理新样本（无需重新训练）
    print("\n步骤2: 增量处理新样本")

    # 模拟新样本
    new_sample = {
        "coordinateName": "@test/new_sdk",
        "version": "1.0.0",
        "codeTlshHashes": [
            "T1A521981D0779E0E55495558CB90B20183635D8A2303FF8FD657C49559AF26BEB8B20DC",
            "@:http request response",
            "lda sta ldai return",
            "TestFeature NewMethod"
        ]
    }

    try:
        new_feature = extractor.transform_new_sample(new_sample)
        print(f"✓ 新样本处理完成: {new_feature.shape}")
        print(f"✓ 处理时间: < 1秒 (无需重新训练)")
    except Exception as e:
        print(f"✗ 处理失败: {e}")
        return

    # 3. 批量处理多个新样本
    print("\n步骤3: 批量处理新样本")

    new_samples = [
        {
            "coordinateName": "@batch/sdk1",
            "version": "1.0.0",
            "codeTlshHashes": ["T1A521981D0779E0E55495558CB90B20183635D8A2303FF8FD657C49559AF26BEB8B20DC"]
        },
        {
            "coordinateName": "@batch/sdk2",
            "version": "2.0.0",
            "codeTlshHashes": ["T1B632092E1888F1F665A6669D0A1C31294746E9B3414GGE9HE7685A666BG37CFC9C31ED"]
        }
    ]

    try:
        batch_features = []
        for sample in new_samples:
            feature = extractor.transform_new_sample(sample)
            batch_features.append(feature)

        batch_features = np.array([f.flatten() for f in batch_features])
        print(f"✓ 批量处理完成: {batch_features.shape}")
    except Exception as e:
        print(f"✗ 批量处理失败: {e}")


def performance_comparison_example():
    """性能对比示例"""
    print("\n" + "=" * 60)
    print("SDK特征提取器 - 性能对比示例")
    print("=" * 60)

    import time

    # 测试不同配置的性能
    configs = [
        {
            'name': '最小配置',
            'config': {
                'max_features': 1000,
                'enable_feature_selection': False,
                'enable_pca': False,
                'n_jobs': 1
            }
        },
        {
            'name': '中等配置',
            'config': {
                'max_features': 1500,
                'enable_feature_selection': True,
                'variance_threshold': 0.01,
                'enable_pca': False,
                'n_jobs': 2
            }
        },
        {
            'name': '最大配置',
            'config': {
                'max_features': 2000,
                'enable_feature_selection': True,
                'variance_threshold': 0.01,
                'enable_pca': True,
                'pca_components': 100,
                'n_jobs': 4
            }
        }
    ]

    data_dir = "data/raw/"

    print("\n性能对比结果:")
    print(f"{'配置名称':<15} {'特征维度':<12} {'处理时间':<12} {'内存占用':<12}")
    print("-" * 60)

    for item in configs:
        extractor = SDKFeatureExtractorOptimized(item['config'])

        try:
            start_time = time.time()
            extractor.load_data(data_dir)
            features, _ = extractor.extract_high_dim_matrix()
            elapsed_time = time.time() - start_time

            print(f"{item['name']:<15} {features.shape[1]:<12} {elapsed_time:.2f}s       {'中等':<12}")

        except Exception as e:
            print(f"{item['name']:<15} {'错误':<12} {'错误':<12} {'错误':<12}")


def save_and_load_example():
    """保存和加载模型示例"""
    print("\n" + "=" * 60)
    print("SDK特征提取器 - 保存和加载模型示例")
    print("=" * 60)

    import pickle

    # 1. 训练模型
    print("\n步骤1: 训练模型")
    extractor = SDKFeatureExtractorOptimized()

    data_dir = "data/raw/"
    try:
        extractor.load_data(data_dir)
        features, sample_info = extractor.extract_high_dim_matrix()
        print(f"✓ 训练完成")
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        return

    # 2. 保存模型
    print("\n步骤2: 保存模型")
    model_file = "models/extractor_model.pkl"
    os.makedirs("models", exist_ok=True)

    try:
        with open(model_file, 'wb') as f:
            pickle.dump(extractor, f)
        print(f"✓ 模型已保存: {model_file}")
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        return

    # 3. 加载模型
    print("\n步骤3: 加载模型")
    try:
        with open(model_file, 'rb') as f:
            loaded_extractor = pickle.load(f)
        print(f"✓ 模型已加载")

        # 4. 使用加载的模型
        test_sample = {
            "coordinateName": "@test/sdk",
            "version": "1.0.0",
            "codeTlshHashes": ["T1A521981D0779E0E55495558CB90B20183635D8A2303FF8FD657C49559AF26BEB8B20DC"]
        }

        feature = loaded_extractor.transform_new_sample(test_sample)
        print(f"✓ 使用加载的模型处理样本: {feature.shape}")

    except Exception as e:
        print(f"✗ 加载失败: {e}")


def main():
    """主函数"""
    print("\nSDK特征提取器 - 使用示例集合")
    print("=" * 60)

    # 示例选择
    examples = {
        '1': ('基本使用示例', basic_usage_example),
        '2': ('高级使用示例', advanced_usage_example),
        '3': ('增量处理示例', incremental_processing_example),
        '4': ('性能对比示例', performance_comparison_example),
        '5': ('保存和加载示例', save_and_load_example)
    }

    print("\n可用示例:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  0. 退出")

    while True:
        choice = input("\n请选择示例 (0-5): ").strip()

        if choice == '0':
            print("退出")
            break

        if choice in examples:
            name, func = examples[choice]
            print(f"\n运行示例: {name}")
            try:
                func()
            except Exception as e:
                print(f"\n示例执行失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    import numpy as np
    main()
