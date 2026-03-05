"""
测试unknown序列记录功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.extractor.sdk_extractor import SDKFeatureExtractor
    print("成功导入SDKFeatureExtractor")
except ImportError as e:
    print(f"导入失败: {e}")
    print("将使用简化版本进行演示")
    SDKFeatureExtractor = None


def test_unknown_sequences_recording():
    """测试unknown序列记录功能"""
    if SDKFeatureExtractor is None:
        print("SDKFeatureExtractor不可用，跳过测试")
        return

    print("=" * 60)
    print("测试unknown序列记录功能")
    print("=" * 60)

    # 创建提取器实例
    extractor = SDKFeatureExtractor()

    # 加载数据
    data_path = "/Users/chengqihan/AICode/sdk_clustering/data/raw"
    try:
        extractor.load_data(data_path)
        print(f"成功加载数据")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return

    # 分析unknown序列
    print("\n分析序列类型分布:")
    print("-" * 60)

    total_unknown = 0
    total_sequences = 0

    for sample in extractor.data:
        extracted = extractor.extract_features_per_sample(sample)
        unknown_seqs = extracted.get('unknown_sequences', [])
        total_sequences += len(sample['codeTlshHashes'])

        if unknown_seqs:
            total_unknown += len(unknown_seqs)
            print(f"\n样本: {extracted['coordinateName']} v{extracted['version']}")
            print(f"  总序列数: {len(sample['codeTlshHashes'])}")
            print(f"  Unknown序列数: {len(unknown_seqs)}")
            print(f"  Unknown序列:")
            for i, seq in enumerate(unknown_seqs[:3]):  # 显示前3个
                display_seq = seq[:60] + '...' if len(seq) > 60 else seq
                print(f"    {i+1}. {display_seq}")
            if len(unknown_seqs) > 3:
                print(f"    ... 还有 {len(unknown_seqs) - 3} 个")

    # 保存unknown序列
    print("\n" + "=" * 60)
    print("保存unknown序列到文件")
    print("=" * 60)

    output_path = "/Users/chengqihan/AICode/sdk_clustering/data/unknown_sequences.json"
    extractor.save_unknown_sequences(output_path)

    # 显示统计信息
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    print(f"总序列数: {total_sequences}")
    print(f"Unknown序列数: {total_unknown}")
    print(f"Unknown比例: {total_unknown/total_sequences*100:.1f}%")

    # 读取保存的文件验证
    try:
        import json
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)

        print(f"\n文件验证:")
        print(f"  文件路径: {output_path}")
        print(f"  包含unknown的样本数: {saved_data['summary']['total_samples_with_unknown']}")
        print(f"  Unknown序列总数: {saved_data['summary']['total_unknown_sequences']}")
        print(f"  Unknown比例: {saved_data['summary']['unknown_rate']}")

    except Exception as e:
        print(f"验证文件失败: {e}")


if __name__ == "__main__":
    print("Unknown序列记录功能测试")
    print("=" * 60)

    test_unknown_sequences_recording()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)