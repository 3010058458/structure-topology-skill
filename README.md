# 结构工程图纸信息提取系统 v5.0

## 项目简介

本项目是一个基于 OCR 和 Vision LLM 的结构工程图纸信息提取系统，采用单张图纸独立处理流程，自动识别图纸类型并提取结构信息。

**核心特性：**
- 🎯 **自动图纸类型识别**：自动判断立面图或平面图
- 🔍 **OCR 增强**：集成 PaddleOCR，提高识别准确率
- 🖼️ **图像预处理**：双图策略（OCR 用预处理图，LLM 用原图）
- 📊 **结构化输出**：生成标准 JSON 格式数据
- 🔄 **智能重试**：LLM 响应解析失败时自动重试（最多 3 次）
- 🛠️ **JSON 修复**：自动修复常见的 JSON 格式错误
- 🛡️ **高容错性**：单张失败不影响其他图片
- 🚀 **易于扩展**：清晰的接口设计，支持自定义客户端
- ✅ **双模型交叉验证**：Gemini 3.1 Pro + Opus 4.6 交叉验证，提高准确率
- 💬 **上下文管理**：支持多轮对话，保存和恢复对话历史

## 版本信息

- **当前版本**：v5.0.0
- **最后更新**：2026-03-04
- **主要更新**：
  - PDF 支持（600 DPI 自动转图，传入 .pdf 文件直接处理）
  - 双模型交叉验证（Gemini 3.1 Pro + Opus 4.6）
  - 两阶段平面图提取：先交叉验证轴网，再注入轴网提取构件
  - 梁坐标后处理：LLM 输出轴线交点标签，系统自动换算数值坐标
  - 大图兼容：自动压缩超出 Anthropic 限制的图片（>3.5MB 或 >8000px）
  - 上下文管理（支持多轮对话）
  - 600 DPI 最高质量图像处理

## 快速开始

### 1. 安装依赖

```bash
# 核心依赖（必需）
pip install requests pillow opencv-python numpy

# OCR 服务依赖（可选，如果使用 OCR 功能）
pip install paddlepaddle paddleocr flask
```

### 2. 配置 API Key

本 Skill 通过 OpenRouter 接入两个模型（Gemini 3.1 Pro Preview + Claude Opus 4.6），
需要配置 OpenRouter API Key。

**方式一：环境变量（推荐）**

```bash
# 主模型 Key（必需）
export OPENROUTER_API_KEY=your-openrouter-api-key

# 交叉验证模型 Key（可选，未设置时自动回退到 OPENROUTER_API_KEY）
export OPUS_API_KEY=your-opus-openrouter-api-key
```

**方式二：config.json 配置文件**

```json
{
  "llm": { "api_key": "your-openrouter-api-key" },
  "cross_validation": { "api_key": "your-opus-openrouter-api-key" }
}
```

> 两个模型可使用同一个 Key，也可分开配置以独立计费。
> 未配置时启动即报错（fail-fast），不会静默失败。

### 3. 启动 OCR 服务（可选）

```bash
cd ocr_service
python ocr_server.py
```

如果不需要 OCR，可以使用 `--no-ocr` 参数跳过此步骤。

### 4. 处理图纸

```bash
cd scripts
# 处理图片文件
python process_drawings.py --images elevation.png 1F.jpg 2F.jpg --output ./output

# 处理 PDF 文件（自动转图再处理）
python process_drawings.py --images drawing.pdf --output ./output
```

## 处理流程

```
用户提交多张图片
    ↓
对每张图片：
    0. PDF 转图（如为 .pdf，600 DPI 转 PNG）
    1. 图像预处理（可选）
       - 转换为灰度图，降噪，对比度增强
       - OCR 使用预处理图，LLM 使用原图
    2. OCR 识别并筛选（置信度 ≥ 0.80）
    2. 双模型交叉验证：识别图纸类型
       - Gemini 3.1 Pro 识别
       - Opus 4.6 识别（大图自动压缩）
       - 比较结果，如有差异则交叉验证
    3. 平面图两阶段提取（立面图单阶段）：
       阶段1 - 交叉验证提取轴网（x_axes / y_axes）
       阶段2 - 注入已知轴网，交叉验证提取构件
       - 梁：LLM 输出 start_grid/end_grid，系统自动换算坐标
       - 支持智能重试（最多 3 次）
    4. 生成独立的 JSON 文件
    5. 保存上下文（支持多轮对话）
    ↓
处理下一张图片
```

## 使用示例

### 命令行使用

```bash
# 处理指定图片（启用交叉验证和上下文管理）
python process_drawings.py --images drawing1.png drawing2.jpg --output ./output

# 处理整个目录
python process_drawings.py --input-dir ../test_images --output ./output

# 禁用 OCR
python process_drawings.py --images drawing.png --no-ocr

# 禁用交叉验证（只使用单个模型，更快）
python process_drawings.py --images drawing.png --no-cross-validation

# 禁用上下文管理
python process_drawings.py --images drawing.png --no-context

# 恢复之前的会话
python process_drawings.py --images drawing.png --session-id 20260303_123456_abc123

```

### Python 脚本使用

```python
from enhanced_image_processor import EnhancedBatchImageProcessor
from client_interfaces import create_ocr_client, create_llm_client
from image_processor import load_config

# 加载配置
config = load_config("config.json")

# 创建客户端
ocr_client = create_ocr_client(config)
llm_client = create_llm_client(config)

# 创建增强处理器（启用交叉验证和上下文管理）
processor = EnhancedBatchImageProcessor(
    ocr_client=ocr_client,
    llm_client=llm_client,
    ocr_confidence_threshold=0.85,
    output_dir="./output",
    cross_validation_enabled=True,  # 启用交叉验证
    context_enabled=True  # 启用上下文管理
)

# 处理图片
image_paths = ["elevation.png", "1F.jpg", "2F.jpg"]
results = processor.process_images(image_paths)

# 查看结果
for result in results:
    print(f"图纸类型: {result.drawing_type}")
    print(f"楼层: {result.floor_id}")
    if "cross_validation" in result.metadata:
        cv = result.metadata["cross_validation"]
        print(f"交叉验证: 一致={cv['consensus_reached']}, 轮数={cv['validation_rounds']}")

# 查看上下文摘要
summary = processor.processor.get_context_summary()
print(f"上下文: {summary['total_messages']} 条消息")
```

## 输出格式

### 立面图输出

```json
{
  "drawing_type": "elevation",
  "floor_id": "立面图",
  "data": {
    "floor_id": "立面图",
    "floor_levels": [
      {"floor": "1F", "elevation": 0.0, "floor_height": 3600.0, "description": "一层楼板面标高"},
      {"floor": "2F", "elevation": 3600.0, "floor_height": 3600.0, "description": "二层楼板面标高"},
      {"floor": "RF", "elevation": 7200.0, "floor_height": null, "description": "屋面标高"}
    ],
    "total_height": 7200.0,
    "floor_count": 2
  },
  "ocr_used": true,
  "metadata": {
    "image_name": "elevation.png",
    "ocr_text_count": 25,
    "type_confidence": 0.95
  }
}
```

### 平面图输出

```json
{
  "drawing_type": "plan",
  "floor_id": "1F",
  "data": {
    "floor_id": "1F",
    "components_above": {
      "columns": [
        {"x": 0, "y": 0, "label": "KZ1", "grid_location": "A-1", "section": "400x400"}
      ],
      "beams": [
        {"start_grid": "A-1", "end_grid": "A-2", "label": "KL1", "section": "250x500",
         "start": [0, 0], "end": [6000, 0]}
      ],
      "walls": [
        {"start": [0, 0], "end": [0, 6000], "thickness": 200, "label": "Q1"}
      ],
      "slabs": []
    },
    "grid_info": {
      "x_axes": [{"label": "1", "coordinate": 0}, {"label": "2", "coordinate": 6000}],
      "y_axes": [{"label": "A", "coordinate": 0}, {"label": "B", "coordinate": 6000}]
    },
    "connection_note": "柱和墙从 1F 楼板延伸至 2F 楼板，梁位于 2F 楼板底面"
  },
  "ocr_used": true,
  "structural_note": "本图纸（1F 平面图）中的构件连接关系：柱和墙从 1F 楼板面向上延伸至上一层楼板面；梁位于上一层楼板底面。具体的 z 坐标需结合立面图的标高信息确定。",
  "metadata": {
    "image_name": "1F.png",
    "ocr_text_count": 42,
    "type_confidence": 0.92
  }
}
```

## 项目结构

```
structure-topology-skill/
├── README.md                     # 项目概述
├── SKILL.md                      # Skill 定义
├── CLAUDE.md                     # Claude Code 执行规范
├── config.json                   # 配置文件
├── requirements.txt              # 依赖列表
├── 启动.bat                      # Windows 快速启动
│
├── scripts/                      # 核心脚本
│   ├── process_drawings.py       # 主入口（支持 PDF/图片）
│   ├── enhanced_image_processor.py  # 增强处理器（交叉验证+上下文）
│   ├── image_processor.py        # 基础处理器（prompt构建、解析）
│   ├── cross_validation.py       # 双模型交叉验证
│   ├── context_manager.py        # 上下文管理
│   ├── client_interfaces.py      # OCR/LLM 客户端（含大图压缩）
│   ├── pdf_to_image.py           # PDF→图片转换（600 DPI）
│   ├── image_preprocessor.py     # 图像预处理（OCR 优化）
│   ├── config_validator.py       # 配置校验
│   └── logger.py                 # 日志工具
│
├── ocr_service/                  # OCR 服务（PaddleOCR）
│   ├── ocr_server.py             # Flask HTTP 服务
│   ├── ocr_cli.py                # 命令行工具
│   └── README.md
│
```

## 配置说明

配置文件 `config.json`：

```json
{
  "project": {
    "version": "5.0.0"
  },
  "ocr": {
    "server_url": "http://localhost:5000",
    "confidence_threshold": 0.80,
    "enabled": true,
    "engine": "PaddleOCR",
    "timeout": 1200
  },
  "llm": {
    "provider": "openrouter",
    "model": "google/gemini-3.1-pro-preview",
    "max_tokens": 4096,
    "temperature": 0.1,
    "reasoning_enabled": true
  },
  "image_preprocessing": {
    "enabled": true,
    "for_ocr_only": true
  }
}
```

**关键配置项：**
- `processing.pdf_dpi`: PDF 转图像的 DPI（默认 600，工程图纸最高质量）
- `ocr.confidence_threshold`: OCR 置信度阈值（0-1），当前 0.80
- `ocr.timeout`: OCR 超时时间（秒），默认 1200 秒
- `llm.api_key`: OpenRouter API Key（优先使用，留空时从环境变量 OPENROUTER_API_KEY 读取）
- `llm.provider`: LLM 提供商（目前仅支持 `openrouter`）
- `llm.model`: LLM 模型名称
- `llm.temperature`: 温度参数（0-1），建议 0.1
- `llm.reasoning_enabled`: 是否启用推理模式（仅 Gemini 3.1 Pro Preview）
- `image_preprocessing.enabled`: 是否启用图像预处理
- `image_preprocessing.for_ocr_only`: 仅对 OCR 使用预处理图，LLM 使用原图（推荐）

## 图像质量优化

本系统针对工程图纸进行了特殊优化，确保最高质量处理：

### 双图策略
- **OCR 识别**：使用预处理图像（灰度化、去噪、对比度增强）
- **LLM 分析**：使用原始高质量图像（保留完整信息）

### 最高 DPI 设置
- **默认 DPI**: 600（工程图纸最高质量）
- **优势**: 确保所有细节清晰可见，包括最小的文字和符号
- **适用**: 所有工程图纸，特别是大尺寸图纸（A0、A1）

### 预处理参数优化
- 去噪强度：h=6（保留细节）
- 对比度增强：clipLimit=1.8（避免过度增强）
- 不使用二值化和锐化（避免信息损失）


## 支持的图片格式

- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- GIF (.gif)
- WebP (.webp)

## 常见问题

### Q: OCR 服务连接失败？
A: 确保 OCR 服务已启动：
```bash
cd ocr_service
python ocr_server.py
```

### Q: LLM API 调用失败？
A: 检查 API Key 是否正确配置：
```bash
# 方式一：环境变量
echo $OPENROUTER_API_KEY

# 方式二：config.json 中的 llm.api_key 字段
```

### Q: 如何禁用 OCR？
A: 使用 `--no-ocr` 参数：
```bash
python process_drawings.py --images drawing.png --no-ocr
```

### Q: 如何更换 LLM 模型？
A: 修改配置文件或使用命令行参数：
```bash
python process_drawings.py --images drawing.png --llm-model "claude-3-5-sonnet-20241022"
```

### Q: 处理失败怎么办？
A: 检查：
1. OCR 服务是否启动
2. API 密钥是否正确设置
3. 图片格式是否支持
4. 网络连接是否正常

## 与旧版本的区别

| 特性 | v4.x | v5.0 |
|------|------|------|
| PDF 支持 | 无 | 自动 600 DPI 转图 |
| 平面图提取 | 单阶段 | 两阶段（轴网→构件）|
| 梁坐标 | LLM 直出数值 | 轴线标签→后处理换算 |
| 大图处理 | 失败（>5MB 报错）| 自动压缩（5MB/8000px 限制）|
| 交叉验证 | 有 | 有（轴网+构件各自验证）|
| 上下文历史 | 全量 | 去重修复（无消息重复）|

## 扩展开发

### 自定义 OCR 客户端

```python
from client_interfaces import OCRClientInterface

class CustomOCRClient(OCRClientInterface):
    def recognize(self, image_path: str):
        # 实现你的 OCR 逻辑
        return [{"text": "...", "confidence": 0.95, "bbox": [...]}]
```

### 自定义 LLM 客户端

```python
from client_interfaces import LLMClientInterface

class CustomLLMClient(LLMClientInterface):
    def chat(self, prompt: str, image_path=None):
        # 实现你的 LLM 调用逻辑
        return "LLM 响应（JSON 格式）"
```

## 文档索引

### 核心文档
- [README.md](README.md) - 项目概述和快速开始
- [SKILL.md](SKILL.md) - Skill 定义文档
- [CLAUDE.md](CLAUDE.md) - Claude Code 执行规范

### 服务文档
- [OCR 服务文档](ocr_service/README.md)

## 许可证

本项目仅供学习和研究使用。

---
