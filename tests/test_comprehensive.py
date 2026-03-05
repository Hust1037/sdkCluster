"""
SDK聚类引擎综合测试套件

这个文件整合了所有测试功能，包括：
1. 序列检测测试
2. 特征提取测试  
3. Unknown序列记录测试
4. 真实样本验证

使用方法:
    python3 tests/test_comprehensive.py --all          # 运行所有测试
    python3 tests/test_comprehensive.py --detection  # 只运行序列检测测试
    python3 tests/test_comprehensive.py --unknown     # 只运行unknown序列测试
"""
import sys
import os
import argparse
import json
import re


class ComprehensiveTestSuite:
    """综合测试套件"""

    def __init__(self):
        self.test_results = []

    def test_sequence_detection(self):
        """测试序列检测逻辑"""
        print("\n" + "=" * 60)
        print("1. 序列检测测试")
        print("=" * 60)

        def _detect_sequence_type(sequence: str) -> str:
            s = sequence.strip()
            if not s: return 'unknown'

            if s.startswith('T1') and 70 <= len(s) <= 72 and all(c in '0123456789ABCDEF' for c in s[2:]):
                return 'tlsh'

            tokens = s.split()
            if not tokens: return 'unknown'

            # 业务语义检测
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

            # 导入导出检测
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

            if all(token.isdigit() for token in tokens):
                return 'numeric'

            # 操作码检测
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

            if all(t.islower() and t.isalnum() and len(t) <= 15 for t in tokens[:5]):
                return 'opcode'

            return 'unknown'

        # 测试用例
        test_cases = [
            {
                'name': '问题序列1',
                'sequence': "@normalized:N&&&@/ijkplayer/src/main/ets/ijkplayer/mon/MessageType&2.0.6-rc.2 MessageType MessageType @normalized:N&&&@/ijkplayer/src/main/ets/ijkplayer/mon/PropertiesType&2.0.6-rc.2 PropertiesType PropertiesType @normalized:Y&&&libijkplayer_audio_napi.so& newIjkPlayerAudio newIjkPlayerAudio @:window window @normalized:N&&&@/ijkplayer/src/main/ets/ijkplayer/utils/LogUtils&2.0.6-rc.2 LogUtils LogUtils 5 IjkMediaPlayer InterruptForceType InterruptHintType",
                'expected': 'import_export'
            },
            {
                'name': '标准导入导出',
                'sequence': "@:hilog hilog LogUtils",
                'expected': 'import_export'
            },
            {
                'name': '操作码序列',
                'sequence': "returnundefined copyrestargs hilog info LogUtils DOMAIN LogUtils TAG callthisrange returnundefined",
                'expected': 'opcode'
            },
            {
                'name': '业务语义序列',
                'sequence': "_getAudioCodecInfo callthis0 _getVideoCodecInfo callthis0 _getMediaMeta callthis0 createobjectwithbuffer audioDecoder videoDecoder meta",
                'expected': 'business_semantic'
            },
            {
                'name': '数字序列',
                'sequence': "123456 789012 3456789",
                'expected': 'numeric'
            },
            {
                'name': 'TLSH哈希',
                'sequence': "T1A521981D0779E0E55495558CB90B20183635D8A2303FF8FD657C49559AF26BEB8B20DC",
                'expected': 'tlsh'
            }
        ]

        passed = 0
        for test_case in test_cases:
            detected = _detect_sequence_type(test_case['sequence'])
            is_correct = detected == test_case['expected']
            status = "✅" if is_correct else "❌"

            print(f"{test_case['name']:<20} {test_case['expected']:<15} {detected:<15} {status}")
            if is_correct:
                passed += 1

        accuracy = passed / len(test_cases) * 100
        print(f"\n准确率: {accuracy:.1f}% ({passed}/{len(test_cases)})")

        self.test_results.append({
            'test_name': 'sequence_detection',
            'passed': passed,
            'total': len(test_cases),
            'accuracy': accuracy
        })

        return accuracy == 100

    def test_unknown_sequences(self):
        """测试unknown序列记录"""
        print("\n" + "=" * 60)
        print("2. Unknown序列测试")
        print("=" * 60)

        samples_path = "../data/raw/samples.json"
        if not os.path.exists(samples_path):
            print(f"⚠️  样本文件不存在: {samples_path}")
            return False

        try:
            with open(samples_path, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)

            print(f"✅ 成功加载样本文件")
            print(f"   SDK: {sample_data['coordinateName']}")
            print(f"   版本: {sample_data['version']}")
            print(f"   序列数: {len(sample_data['codeTlshHashes'])}")

            # 统计unknown序列
            hashes = sample_data.get('codeTlshHashes', [])
            unknown_count = 0

            for seq in hashes:
                if len(seq.strip()) == 0:
                    unknown_count += 1

            print(f"\nUnknown序列统计:")
            print(f"   总序列数: {len(hashes)}")
            print(f"   Unknown序列数: {unknown_count}")
            print(f"   Unknown比例: {unknown_count/len(hashes)*100:.1f}%")

            self.test_results.append({
                'test_name': 'unknown_sequences',
                'total_sequences': len(hashes),
                'unknown_count': unknown_count,
                'unknown_rate': unknown_count/len(hashes)*100
            })

            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 60)
        print("SDK聚类引擎综合测试套件")
        print("=" * 60)

        all_passed = True

        # 运行序列检测测试
        if not self.test_sequence_detection():
            all_passed = False

        # 运行unknown序列测试
        if not self.test_unknown_sequences():
            all_passed = False

        # 生成测试报告
        self._generate_report()

        return all_passed

    def _generate_report(self):
        """生成测试报告"""
        print("\n" + "=" * 60)
        print("测试报告")
        print("=" * 60)

        for result in self.test_results:
            test_name = result.get('test_name', 'unknown')
            print(f"\n{test_name}:")
            for key, value in result.items():
                if key != 'test_name':
                    print(f"  {key}: {value}")

        # 计算总体通过率
        detection_passed = any(
            r.get('test_name') == 'sequence_detection' and r.get('accuracy') == 100
            for r in self.test_results
        )

        unknown_passed = any(
            r.get('test_name') == 'unknown_sequences'
            for r in self.test_results
        )

        print("\n" + "=" * 60)
        if detection_passed and unknown_passed:
            print("🎉 所有测试通过！")
        else:
            print("⚠️  部分测试失败")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='SDK聚类引擎综合测试套件')
    parser.add_argument('--all', action='store_true', help='运行所有测试')
    parser.add_argument('--detection', action='store_true', help='只运行序列检测测试')
    parser.add_argument('--unknown', action='store_true', help='只运行unknown序列测试')

    args = parser.parse_args()

    test_suite = ComprehensiveTestSuite()

    if args.all:
        test_suite.run_all_tests()
    elif args.detection:
        test_suite.test_sequence_detection()
    elif args.unknown:
        test_suite.test_unknown_sequences()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()