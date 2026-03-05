"""
简单测试unknown序列记录功能（不依赖外部库）
"""
import json
import re


def _detect_sequence_type(sequence: str) -> str:
    """
    从sdk_extractor.py中复制的检测方法
    """
    s = sequence.strip()
    if not s: return 'unknown'

    # 1. TLSH检测
    if s.startswith('T1') and 70 <= len(s) <= 72 and all(c in '0123456789ABCDEF' for c in s[2:]):
        return 'tlsh'

    tokens = s.split()
    if not tokens: return 'unknown'

    # 2. 业务语义检测
    business_semantic_indicators = 0
    if re.search(r'_[gs]et[A-Z][a-zA-Z]*', s):
        business_semantic_indicators += 2
    if re.search(r'mOn[A-Z][a-zA-Z]+', s):
        business_semantic_indicators += 2
    business_keywords = ['PROP_', 'FFP_', 'Listener', 'Handler']
    if any(keyword in s for keyword in business_keywords):
        business_semantic_indicators += 2
    business_functions = ['definefieldbyname', 'createobjectwithbuffer', 'getInstance']
    if any(func in s for func in business_functions):
        business_semantic_indicators += 2
    if re.search(r'[A-Z][a-zA-Z]+&[0-9]+\.[0-9]+', s):
        business_semantic_indicators += 1

    if business_semantic_indicators >= 2:
        return 'business_semantic'

    # 3. 导入导出检测
    import_export_indicators = 0
    if '@:' in s: import_export_indicators += 3
    if '@normalized' in s: import_export_indicators += 2
    if re.search(r'@/[a-zA-Z0-9_./-]+', s): import_export_indicators += 2
    if re.search(r'[a-zA-Z0-9_]+\.so', s): import_export_indicators += 2
    import_keywords = ['import', 'export', 'require', 'declare', '.webview', 'router']
    if any(keyword in s.lower() for keyword in import_keywords):
        import_export_indicators += 2
    if '&&&' in s: import_export_indicators += 2
    if re.search(r'&[0-9]+\.[0-9]+', s): import_export_indicators += 1

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
        if opcode_count >= 3:
            return 'opcode'

    # 6. 兜底：如果包含小写字母为主的token，判定为操作码
    if all(t.islower() and t.isalnum() and len(t) <= 15 for t in tokens[:5]):
        return 'opcode'

    return 'unknown'


def save_unknown_sequences(samples_data, output_path: str = 'unknown_sequences.json'):
    """
    保存无法识别的unknown序列到JSON文件
    """
    unknown_data = []

    for sample in samples_data:
        hashes = sample.get('codeTlshHashes', [])
        unknown_seqs = []

        for seq in hashes:
            seq_type = _detect_sequence_type(seq)
            if seq_type == 'unknown':
                unknown_seqs.append(seq)

        if unknown_seqs:
            unknown_data.append({
                'coordinateName': sample.get('coordinateName', ''),
                'version': sample.get('version', ''),
                'unknown_count': len(unknown_seqs),
                'unknown_sequences': unknown_seqs,
                'total_sequences': len(hashes)
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
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output


def test_unknown_recording():
    """测试unknown序列记录功能"""
    print("=" * 60)
    print("测试unknown序列记录功能")
    print("=" * 60)

    # 读取samples.json
    samples_path = "/Users/chengqihan/AICode/sdk_clustering/data/raw/samples.json"
    try:
        with open(samples_path, 'r', encoding='utf-8') as f:
            samples_data = json.load(f)
        print(f"成功加载samples.json")
        print(f"SDK名称: {samples_data['coordinateName']}")
        print(f"版本: {samples_data['version']}")
        print(f"序列数量: {len(samples_data['codeTlshHashes'])}")
    except Exception as e:
        print(f"加载samples.json失败: {e}")
        return

    # 分析unknown序列
    print("\n分析unknown序列:")
    print("-" * 60)

    hashes = samples_data.get('codeTlshHashes', [])
    unknown_seqs = []

    for seq in hashes:
        seq_type = _detect_sequence_type(seq)
        if seq_type == 'unknown':
            unknown_seqs.append(seq)

    print(f"找到 {len(unknown_seqs)} 个unknown序列:")
    for i, seq in enumerate(unknown_seqs):
        display_seq = seq[:60] + '...' if len(seq) > 60 else seq
        print(f"  {i+1}. {display_seq}")

    # 保存unknown序列
    output_path = "/Users/chengqihan/AICode/sdk_clustering/data/unknown_sequences.json"

    print("\n" + "=" * 60)
    print("保存unknown序列到文件")
    print("=" * 60)

    result = save_unknown_sequences([samples_data], output_path)

    print(f"已保存unknown序列到文件: {output_path}")
    print(f"  - 包含unknown的样本数: {result['summary']['total_samples_with_unknown']}")
    print(f"  - Unknown序列总数: {result['summary']['total_unknown_sequences']}")
    print(f"  - Unknown比例: {result['summary']['unknown_rate']}")

    # 验证文件
    print("\n" + "=" * 60)
    print("验证保存的文件")
    print("=" * 60)

    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)

        print(f"✅ 文件验证成功")
        print(f"  文件大小: {len(saved_data)} 个顶层键")
        print(f"  包含summary: {'summary' in saved_data}")
        print(f"  包含unknown_data: {'unknown_data' in saved_data}")

        if 'unknown_data' in saved_data:
            unknown_data = saved_data['unknown_data']
            if unknown_data:
                first_unknown = unknown_data[0]
                print(f"  第一个unknown样本: {first_unknown['coordinateName']}")
                print(f"  该样本的unknown数量: {first_unknown['unknown_count']}")
                if first_unknown['unknown_sequences']:
                    print(f"  第一个unknown序列: {first_unknown['unknown_sequences'][0][:50]}...")

    except Exception as e:
        print(f"❌ 文件验证失败: {e}")


if __name__ == "__main__":
    print("Unknown序列记录功能测试")
    print("=" * 60)

    test_unknown_recording()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)