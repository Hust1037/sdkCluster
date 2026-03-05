"""
测试操作码检测修复
"""
import re


def _detect_sequence_type(sequence: str) -> str:
    """修复后的检测方法"""
    s = sequence.strip()
    if not s:
        return 'unknown'

    # 1. TLSH检测
    if s.startswith('T1') and 70 <= len(s) <= 72 and all(c in '0123456789ABCDEF' for c in s[2:]):
        return 'tlsh'

    tokens = s.split()
    if not tokens:
        return 'unknown'

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
    if '@:' in s:
        import_export_indicators += 3
    if '@normalized' in s:
        import_export_indicators += 2
    if re.search(r'@/[a-zA-Z0-9_./-]+', s):
        import_export_indicators += 2
    if re.search(r'[a-zA-Z0-9_]+\.so', s):
        import_export_indicators += 2
    import_keywords = ['import', 'export', 'require', 'declare', '.webview', 'router']
    if any(keyword in s.lower() for keyword in import_keywords):
        import_export_indicators += 2
    if '&&&' in s:
        import_export_indicators += 2
    if re.search(r'&[0-9]+\.[0-9]+', s):
        import_export_indicators += 1

    if import_export_indicators >= 2:
        return 'import_export'

    # 4. 数字序列检测
    if all(token.isdigit() for token in tokens):
        return 'numeric'

    # 5. 操作码检测（修复后）
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


def test_opcode_detection():
    """测试操作码检测"""
    print("=" * 60)
    print("测试操作码检测修复")
    print("=" * 60)

    # 之前被误判为unknown的序列
    test_cases = [
        {
            'sequence': "mLogOpen returnundefined ",
            'expected': 'opcode',
            'reason': '包含returnundefined操作码'
        },
        {
            'sequence': "ldfalse mLogOpen ",
            'expected': 'opcode',
            'reason': '包含ldfalse操作码'
        },
        {
            'sequence': "isfalse setWindowKeepScreenOn returnundefined ",
            'expected': 'opcode',
            'reason': '包含isfalse和returnundefined操作码'
        },
        {
            'sequence': "mVideoWidth ",
            'expected': 'opcode',
            'reason': '属性访问模式 m前缀+大写字母'
        },
        {
            'sequence': "mVideoHeight ",
            'expected': 'opcode',
            'reason': '属性访问模式 m前缀+大写字母'
        },
        {
            'sequence': "newIjkPlayerAudio ldfalse newobjrange ijkplayer_audio_napi id returnundefined ",
            'expected': 'opcode',
            'reason': '包含ldfalse、newobjrange、returnundefined操作码'
        },
        {
            'sequence': "mVideoSarDen ",
            'expected': 'opcode',
            'reason': '属性访问模式 m前缀+大写字母'
        },
        {
            'sequence': "ijkplayer_napi isfalse id returnundefined ",
            'expected': 'opcode',
            'reason': '包含isfalse和returnundefined操作码'
        },
        {
            'sequence': "mVideoSarNum ",
            'expected': 'opcode',
            'reason': '属性访问模式 m前缀+大写字母'
        },
        {
            'sequence': "ldhole defineclasswithbuffer 0 prototype definemethod callthis0 stmodulevar returnundefined ",
            'expected': 'opcode',
            'reason': '包含多个操作码'
        }
    ]

    passed = 0
    failed = []

    for test in test_cases:
        detected = _detect_sequence_type(test['sequence'])
        is_correct = detected == test['expected']
        status = "✅" if is_correct else "❌"

        print(f"\n序列: {test['sequence'][:50]}...")
        print(f"  期望: {test['expected']}")
        print(f"  实际: {detected}")
        print(f"  说明: {test['reason']}")
        print(f"  结果: {status}")

        if is_correct:
            passed += 1
        else:
            failed.append(test)

    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{len(test_cases)} 通过")
    print(f"准确率: {passed/len(test_cases)*100:.1f}%")
    print("=" * 60)

    if failed:
        print("\n失败的测试用例:")
        for test in failed:
            print(f"  - {test['sequence'][:40]}... (期望: {test['expected']})")

    return passed == len(test_cases)


if __name__ == "__main__":
    test_opcode_detection()
