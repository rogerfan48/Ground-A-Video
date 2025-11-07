# convert_dataset.py 使用说明

## 配置说明

### 1. 全局默认配置 (DEFAULT_CONFIG)

位于脚本顶部的 CONFIG SECTION，适用于所有未单独配置的样本：

```python
DEFAULT_CONFIG = {
    'n_sample_frames': 8,        # 要采样的总帧数
    'start_sample_frame': 0,      # 开始采样的帧索引 (0-based)
    'sampling_rate': 1,           # 采样率：每N帧取一帧
}
```

### 2. 每个样本的自定义配置 (SAMPLE_CONFIGS)

为特定样本设置不同的参数：

```python
SAMPLE_CONFIGS = {
    'o2o/3ball': {
        'n_sample_frames': 15,
        'start_sample_frame': 10,
        'sampling_rate': 4,
    },
}
```

### 3. Assets 目录配置

```python
ASSETS_DIR = 'assets'  # 可以是绝对路径或相对路径
```

## 参数说明

- **n_sample_frames**: 总共要取的帧数
- **start_sample_frame**: 从第几帧开始取（0-based，对应 frames/ 目录中的文件名）
- **sampling_rate**: 采样间隔，每隔多少帧取一帧

## 使用示例

### 示例 1: 使用默认配置处理所有样本

```python
DEFAULT_CONFIG = {
    'n_sample_frames': 8,
    'start_sample_frame': 0,
    'sampling_rate': 1,
}

SAMPLE_CONFIGS = {}  # 空字典，所有样本使用默认配置
```

运行后，所有样本都会从第0帧开始，每隔1帧取一帧，共取8帧：
- 采样帧: [0, 1, 2, 3, 4, 5, 6, 7]

### 示例 2: 为特定样本自定义配置

```python
DEFAULT_CONFIG = {
    'n_sample_frames': 8,
    'start_sample_frame': 0,
    'sampling_rate': 1,
}

SAMPLE_CONFIGS = {
    'o2o/3ball': {
        'n_sample_frames': 15,
        'start_sample_frame': 10,
        'sampling_rate': 4,
    },
    'o2o/dog-cat': {
        'n_sample_frames': 10,
        'start_sample_frame': 5,
        'sampling_rate': 3,
    },
}
```

结果：
- **3ball**: 从第10帧开始，每隔4帧取一帧，共取15帧
  - 采样帧: [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66]
- **dog-cat**: 从第5帧开始，每隔3帧取一帧，共取10帧
  - 采样帧: [5, 8, 11, 14, 17, 20, 23, 26, 29, 32]
- **其他样本**: 使用默认配置
  - 采样帧: [0, 1, 2, 3, 4, 5, 6, 7]

### 示例 3: 自定义 Assets 目录

```python
# 相对路径
ASSETS_DIR = 'assets'

# 或绝对路径
ASSETS_DIR = '/home/roger/code/TRACK/Ground-A-Video/assets'
```

## 输出

### 生成的文件结构

```
Ground-A-Video/
├── video_images/
│   └── o2o/3ball/
│       ├── 1.jpg      # 对应原始的第10帧
│       ├── 2.jpg      # 对应原始的第14帧
│       ├── 3.jpg      # 对应原始的第18帧
│       └── ...
└── video_configs/
    └── o2o/3ball.yaml  # 包含15个frames的配置
```

### YAML 配置文件

生成的 YAML 文件会自动包含正确数量的 phrases 和 locations：

```yaml
ckpt: "gligen-inpainting-text-box/diffusion_pytorch_model.bin"
input_images_path: "video_images/o2o/3ball"
prompt: "TODO: Add target prompt for 3ball"
source_prompt: "TODO: Add source prompt for 3ball"
phrases:
  - ['obj1', 'obj2', 'obj3']
  - ['obj1', 'obj2', 'obj3']
  # ... (共 n_sample_frames 行)
locations:
  - [[0.04, 0.41, 0.16, 0.52], [0.24, 0.46, 0.44, 0.58], ...]
  # ... (共 n_sample_frames 行)
```

## 运行脚本

```bash
python3 convert_dataset.py
```

脚本会显示每个样本的配置信息：

```
📹 Processing o2o/3ball: 86 frames
   Config: n_sample_frames=15, start_frame=10, rate=4
   Sampled frame indices: [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66]
   Found 3 objects: ['obj1', 'obj2', 'obj3']
   ✅ Successfully processed o2o/3ball
```

## 注意事项

1. **帧索引越界**: 如果 `start_sample_frame + (n_sample_frames - 1) * sampling_rate >= total_frames`，脚本会发出警告并只采样可用的帧。

2. **配置格式**: `SAMPLE_CONFIGS` 的 key 必须是 `'category/sample_name'` 格式，例如 `'o2o/3ball'`。

3. **输出文件命名**: 输出的图片文件名始终是 `1.jpg, 2.jpg, 3.jpg, ...`，与原始帧索引无关。

4. **Mask 转换**: 脚本会自动将每个采样帧对应的 mask 转换为 bounding box 坐标。
