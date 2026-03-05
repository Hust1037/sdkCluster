# Unknown序列记录功能使用说明

## 功能概述

新增的`save_unknown_sequences()`方法可以将无法识别的"unknown"序列保存到JSON文件中，便于后续分析和优化序列检测逻辑。

## 修改的文件

1. **`src/extractor/sdk_extractor.py`** - 原始版本
2. **`src/extractor/sdk_extractor_optimized.py`** - 优化版本

## 主要改动

### 1. 修改了`extract_features_per_sample()`方法
- 新增`unknown_sequences`字段，用于存储无法识别的序列
- 当序列类型为'unknown'时，将其添加到`unknown_sequences`列表

### 2. 新增`save_unknown_sequences()`方法
```python
def save_unknown_sequences(self, output_path: str = 'unknown_sequences.json'):
    """
    保存无法识别的unknown序列到JSON文件
    用于后续分析和优化序列检测逻辑
    """
```

## 输出格式

生成的JSON文件包含以下结构：

```json
{
  "summary": {
    "total_samples_with_unknown": 1,
    "total_unknown_sequences": 13,
    "unknown_rate": "1300.0%"
  },
  "unknown_data": [
    {
      "coordinateName": "@ohos/ijkplayer",
      "version": "2.0.6-rc.2",
      "unknown_count": 13,
      "unknown_sequences": [
        "mLogOpen returnundefined ",
        "ldfalse mLogOpen ",
        "isfalse setWindowKeepScreenOn returnundefined ",
        "mVideoWidth ",
        "mVideoHeight ",
        "newIjkPlayerAudio ldfalse newobjrange ijkplayer_audio_napi id returnundefined ",
        "mVideoSarDen ",
        "ijkplayer_napi isfalse id returnundefined ",
        "mVideoSarNum ",
        "MessageType",
        "ldhole defineclasswithbuffer 0 prototype definemethod callthis0 stmodulevar returnundefined ",
        "PropertiesType",
        "ldhole defineclasswithbuffer 0 prototype definemethod callthis0 stmodulevar returnundefined "
      ],
      "total_sequences": 109
    }
  ]
}
```

## 使用方法

### 方法一：使用SDKFeatureExtractor类

```python
from src.extractor.sdk_extractor import SDKFeatureExtractor

# 创建提取器实例
extractor = SDKFeatureExtractor()

# 加载数据
extractor.load_data("path/to/data/directory")

# 提取特征
matrix, sample_info = extractor.extract_high_dim_matrix()

# 保存unknown序列
extractor.save_unknown_sequences("path/to/output/unknown_sequences.json")
```

### 方法二：使用测试脚本

#### 简单测试（不依赖外部库）
```bash
python3 tests/test_unknown_recording.py
```

#### 完整测试（需要numpy等依赖）
```bash
python3 examples/test_unknown_sequences.py
```

## 示例输出

运行测试脚本后的输出示例：

```
============================================================
测试unknown序列记录功能
============================================================
成功加载samples.json
SDK名称: @ohos/ijkplayer
版本: 2.0.6-rc.2
序列数量: 109

分析unknown序列:
------------------------------------------------------------
找到 13 个unknown序列:
  1. mLogOpen returnundefined 
  2. ldfalse mLogOpen 
  3. isfalse setWindowKeepScreenOn returnundefined 
  4. mVideoWidth 
  5. mVideoHeight 
  6. newIjkPlayerAudio ldfalse newobjrange ijkplayer_audio_napi i...
  7. mVideoSarDen 
  8. ijkplayer_napi isfalse id returnundefined 
  9. mVideoSarNum 
  10. MessageType
  11. ldhole defineclasswithbuffer 0 prototype definemethod callth...
  12. PropertiesType
  13. ldhole defineclasswithbuffer 0 prototype definemethod callth...

============================================================
保存unknown序列到文件
============================================================
已保存unknown序列到文件: /path/to/unknown_sequences.json
  - 包含unknown的样本数: 1
  - Unknown序列总数: 13
  - Unknown比例: 1300.0%
```

## 文件说明

### unknown_sequences.json

包含以下信息：

1. **summary** - 统计摘要
   - `total_samples_with_unknown`: 包含unknown序列的样本数
   - `total_unknown_sequences`: unknown序列总数
   - `unknown_rate`: unknown序列比例

2. **unknown_data** - 详细的unknown序列数据
   - 每个包含unknown序列的样本记录
   - 包含SDK名称、版本、unknown序列列表等

## 分析建议

根据unknown序列的分析，可以优化序列检测逻辑：

### 常见的unknown序列类型

1. **属性访问序列**
   ```python
   "mLogOpen returnundefined "
   "mVideoWidth "
   "mVideoHeight "
   ```
   - 建议：添加`m[A-Z][a-zA-Z]*`模式检测

2. **类型名称序列**
   ```python
   "MessageType"
   "PropertiesType"
   ```
   - 建议：添加`[A-Z][a-zA-Z]*Type`模式检测

3. **混合操作码序列**
   ```python
   "newIjkPlayerAudio ldfalse newobjrange ijkplayer_audio_napi id returnundefined "
   ```
   - 建议：这类序列可能需要更复杂的分析

### 优化建议

根据分析，可以进一步增强`_detect_sequence_type()`方法：

```python
# 添加属性访问检测
if re.search(r'm[A-Z][a-zA-Z]+', s):
    return 'attribute_access'

# 添加类型名称检测
if re.search(r'[A-Z][a-zA-Z]*Type', s):
    return 'type_name'

# 添加对象创建检测
if re.search(r'new[A-Z][a-zA-Z]+', s):
    return 'object_creation'
```

## 测试文件

- **`tests/test_unknown_recording.py`** - 简单测试（不依赖外部库）
- **`examples/test_unknown_sequences.py`** - 完整测试（需要依赖库）

## 注意事项

1. **文件覆盖**: 每次调用`save_unknown_sequences()`都会覆盖之前的文件
2. **编码格式**: 输出文件使用UTF-8编码
3. **路径处理**: 请确保输出路径具有写入权限
4. **内存使用**: 对于大量样本，unknown序列可能占用较多内存

## 后续优化方向

1. **分析unknown序列模式**: 根据unknown序列的特征，添加新的检测规则
2. **用户自定义规则**: 允许用户提供自定义的序列检测规则
3. **机器学习分类**: 使用ML算法自动学习序列类型
4. **可视化工具**: 提供unknown序列的可视化分析工具

## 示例：手动分析unknown序列

```python
import json

# 读取unknown序列文件
with open('unknown_sequences.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 分析每个unknown序列
for sample in data['unknown_data']:
    print(f"样本: {sample['coordinateName']}")
    print(f"Unknown序列数: {sample['unknown_count']}")

    for i, seq in enumerate(sample['unknown_sequences']):
        print(f"  {i+1}. {seq[:50]}...")

        # 分析特征
        tokens = seq.split()
        print(f"     Token数: {len(tokens)}")
        print(f"     长度: {len(seq)}")
        print(f"     包含大写: {any(c.isupper() for c in seq)}")
        print(f"     包含数字: {any(c.isdigit() for c in seq)}")
```

---

**功能实现时间**: 2026-03-05
**适用版本**: sdk_extractor.py, sdk_extractor_optimized.py
**测试状态**: ✅ 已通过验证