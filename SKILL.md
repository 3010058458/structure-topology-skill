---
name: structure-topology-reconstruction
version: 5.0.0
description: 从结构工程图纸（PDF / 图片）自动提取楼层标高与构件信息，输出标准化 JSON，可选生成三维 VTU 模型。采用 OCR + 双模型交叉验证（Gemini 3.1 Pro + Claude Opus 4.6）流水线，适用于建筑结构信息化场景。
---

# 结构工程图纸信息提取 Skill

## 目录

- [概述](#概述)
- [支持的图纸类型与格式](#支持的图纸类型与格式)
- [前置条件](#前置条件)
- [API Key 配置](#api-key-配置)
- [操作步骤](#操作步骤)
- [输出格式规范](#输出格式规范)
- [命令行参考](#命令行参考)
- [错误处理策略](#错误处理策略)
- [性能基准](#性能基准)
- [已知限制](#已知限制)
- [资源索引](#资源索引)

---

## 概述

本 Skill 实现了一套面向结构工程图纸的自动化信息提取流水线：

```
输入（PDF / 图片）
  │
  ├─ [可选] PDF → PNG 转换（600 DPI）
  ├─ [可选] 图像预处理（去噪 / 增强对比度，仅供 OCR 使用）
  │
  ▼
  OCR 识别（PaddleOCR，置信度 >= 0.80）
  │
  ▼
  图纸类型识别（elevation / plan）
  │         ↑
  └── 双模型交叉验证（Gemini 3.1 Pro + Claude Opus 4.6）
  │
  ▼
  结构信息提取
  │   ├─ 立面图：单阶段 → 楼层标高数组
  │   └─ 平面图：两阶段
  │         阶段 1：交叉验证提取轴网（x_axes / y_axes）
  │         阶段 2：注入轴网，交叉验证提取构件（柱 / 梁 / 墙）
  │
  ▼
  输出 JSON 文件（per-image）
  │
  ▼
  [可选] JSON → VTU 三维模型（ParaView 可视化）
```

**核心能力：**
- 自动识别立面图 / 平面图，无需人工标注
- OCR 与 Vision LLM 双通道，标高、轴网标注识别准确率高
- 双模型交叉验证，自动消解歧义，最多 3 轮收敛
- 大图自动处理：超过 Anthropic 5 MB / 8000 px 限制时自动压缩
- OCR 服务 500 错误自动重启恢复，无需人工干预
- 单张图纸失败不影响批量任务其余图纸

---

## 支持的图纸类型与格式

| 维度 | 支持范围 |
|------|----------|
| 图纸类型 | 立面图（elevation）、结构平面图（plan）|
| 文件格式 | PDF、PNG、JPG/JPEG、BMP、TIFF/TIF、GIF、WebP |
| 图纸尺寸 | A0 至 A4，任意 DPI（超限自动缩放）|
| 标注语言 | 中文（含混合英文字母轴号）|

---

## 前置条件

### 依赖安装

```bash
# 核心依赖（必需）
pip install requests>=2.28.0 pillow>=10.0.0 opencv-python>=4.8.0 numpy>=1.24.0

# PDF 处理（必需，处理 PDF 输入时）
pip install pymupdf>=1.23.0

# OCR 服务（可选，禁用时仅用 Vision LLM，精度略低）
pip install paddlepaddle>=3.0.0 paddleocr>=2.7.0 flask>=3.0.0

# 三维重建（可选，生成 VTU 文件时）
pip install pyvista>=0.43.0
```

### OCR 服务启动（可选）

```bash
# 启动 OCR 服务（首次启动需等待约 15 秒完成模型加载）
cd ocr_service
python ocr_server.py

# 健康检查
curl http://localhost:5000/health
# 正常返回：{"status": "ok", "message": "OCR service is running"}
```

> **500 错误自动恢复**：识别过程中若 OCR 服务返回 HTTP 500，`PaddleOCRClient`
> 会自动终止旧进程、重启服务并重试，无需手动干预。
> 详见 `scripts/client_interfaces.py` -> `_restart_service()`。

---

## API Key 配置

本 Skill 通过 [OpenRouter](https://openrouter.ai) 接入两个模型：

| 模型 | 用途 | OpenRouter Model ID |
|------|------|---------------------|
| Google Gemini 3.1 Pro Preview | 主推理模型 | `google/gemini-3.1-pro-preview` |
| Anthropic Claude Opus 4.6 | 交叉验证模型 | `anthropic/claude-opus-4.6` |

### 配置方式（优先级由高到低）

**方式一：环境变量（推荐，适合生产部署）**

```bash
# 主模型 Key（必需）
export OPENROUTER_API_KEY=<your-openrouter-api-key>

# 交叉验证模型 Key（可选，未设置时自动回退到 OPENROUTER_API_KEY）
export OPUS_API_KEY=<your-opus-openrouter-api-key>
```

**方式二：config.json 配置文件**

```json
{
  "llm": {
    "api_key": "<your-openrouter-api-key>"
  },
  "cross_validation": {
    "api_key": "<your-opus-openrouter-api-key>"
  }
}
```

**方式三：禁用交叉验证（仅需一个 Key）**

```bash
python process_drawings.py --images drawing.png --no-cross-validation
```

> **Key 共用**：两个模型可使用同一个 OpenRouter 账号的同一个 Key，也可分开配置以独立计费。
> 所有 Key 均不写入代码，未配置时启动即报错（fail-fast），避免运行时静默失败。

---

## 操作步骤

### 步骤 0（可选）：PDF 转图像

`process_drawings.py` 主入口会自动识别并转换 PDF，无需手动操作。
如需手动调用：

```python
from pdf_to_image import PDFToImageConverter

converter = PDFToImageConverter(dpi=600, output_format="png")
image_paths = converter.convert_pdf_to_images(
    pdf_path="drawings.pdf",
    output_dir="./temp_images"
)
# 返回每页对应的图片路径列表
```

### 步骤 1：OCR 识别

- 调用 PaddleOCR HTTP 服务识别图片中的文字
- 筛选置信度 >= 0.80 的结果，构建文字摘要
- 图像预处理（灰度化、去噪、对比度增强）仅对 OCR 使用，LLM 始终收到原图

### 步骤 2：图纸类型识别（双模型交叉验证）

Gemini 和 Opus 各自独立判断，比较 `drawing_type` 字段：
- 一致 -> 直接通过
- 不一致 -> 互相参考对方结果，最多 3 轮，最终以合并策略兜底

判断依据：
- **立面图**：竖向视图，含楼层标高符号（倒三角）、楼层线
- **平面图**：俯视图，含轴网（数字轴 + 字母轴）、柱梁墙平面布置

### 步骤 3：结构信息提取

**立面图（单阶段）：**

```
交叉验证提取 → floor_levels 数组（楼层名、标高、层高）
```

**平面图（两阶段）：**

```
阶段 1：交叉验证专项提取轴网 → x_axes / y_axes（轴线标签 + 坐标）
阶段 2：将已知轴网注入 Prompt → 交叉验证提取构件
        梁：LLM 输出 start_grid / end_grid → 系统自动换算数值坐标
```

### 步骤 4：保存 JSON 文件

输出路径：`{output_dir}/{image_name}_extraction.json`

---

## 输出格式规范

### 立面图

```json
{
  "drawing_type": "elevation",
  "floor_id": "立面图",
  "data": {
    "floor_id": "立面图",
    "floor_levels": [
      {"floor": "1F", "elevation": 0.0,    "floor_height": 3600.0, "description": "一层楼板面标高"},
      {"floor": "2F", "elevation": 3600.0, "floor_height": 3600.0, "description": "二层楼板面标高"},
      {"floor": "RF", "elevation": 7200.0, "floor_height": null,   "description": "屋面标高"}
    ],
    "total_height": 7200.0,
    "floor_count": 2
  },
  "ocr_used": true,
  "metadata": {
    "image_name": "elevation.png",
    "ocr_text_count": 25,
    "type_confidence": 0.95,
    "cross_validation": {
      "consensus_reached": true,
      "validation_rounds": 1,
      "has_differences": false,
      "differences_count": 0
    }
  }
}
```

### 平面图

```json
{
  "drawing_type": "plan",
  "floor_id": "1F",
  "data": {
    "floor_id": "1F",
    "components_above": {
      "columns": [
        {"x": 0,    "y": 0, "label": "KZ1", "grid_location": "A-1", "section": "400x400"},
        {"x": 6000, "y": 0, "label": "KZ1", "grid_location": "A-2", "section": "400x400"}
      ],
      "beams": [
        {
          "start_grid": "A-1", "end_grid": "A-2",
          "start": [0, 0], "end": [6000, 0],
          "label": "KL1", "section": "250x500"
        }
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
  "metadata": {
    "image_name": "1F.png",
    "ocr_text_count": 42,
    "type_confidence": 0.92
  }
}
```

**坐标系：**
- 原点：1 轴与 A 轴交点
- X 轴：数字轴增大方向（1 -> 2 -> 3...）
- Y 轴：字母轴增大方向（A -> B -> C...）
- 单位：毫米（mm）

**构件连接语义（N 层平面图）：**
- 柱 / 剪力墙：底部在 N 层楼板面，顶部在 N+1 层楼板面
- 梁：位于 N+1 层楼板底面

---

## 命令行参考

```bash
cd scripts

# 基础用法
python process_drawings.py --images elevation.png 1F.jpg 2F.jpg --output ./output

# 处理 PDF（自动拆页转图）
python process_drawings.py --images drawings.pdf --output ./output

# 扫描整个目录
python process_drawings.py --input-dir ../drawings --output ./output

# 禁用 OCR（仅 Vision LLM）
python process_drawings.py --images drawing.png --no-ocr

# 禁用交叉验证（加速，只需一个 API Key）
python process_drawings.py --images drawing.png --no-cross-validation

# 禁用上下文管理
python process_drawings.py --images drawing.png --no-context

# 恢复之前的会话
python process_drawings.py --images drawing.png --session-id 20260303_123456_abc123

# 指定模型
python process_drawings.py --images drawing.png --llm-model "google/gemini-3.1-pro-preview"
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--images` | — | 图片 / PDF 路径列表（与 `--input-dir` 二选一）|
| `--input-dir` | — | 自动扫描目录中所有图片 |
| `--output` | `./output` | JSON 输出目录 |
| `--config` | `../config.json` | 配置文件路径 |
| `--no-ocr` | off | 禁用 OCR，仅用 Vision LLM |
| `--no-cross-validation` | off | 禁用双模型交叉验证 |
| `--no-context` | off | 禁用多轮对话上下文管理 |
| `--session-id` | 自动生成 | 指定会话 ID 以恢复历史上下文 |
| `--llm-provider` | `openrouter` | LLM 提供商 |
| `--llm-model` | config 中值 | 覆盖主模型名称 |
| `--ocr-threshold` | config 中值 | OCR 置信度阈值（0-1）|

---

## 错误处理策略

| 场景 | 处理方式 |
|------|----------|
| OCR 服务未启动 | 跳过 OCR，继续用 Vision LLM 降级处理 |
| OCR 返回 HTTP 500 | 自动重启服务、等待 15 秒、重试一次 |
| LLM 返回非 JSON | 自动重试最多 3 次，仍失败则跳过该图纸 |
| 图纸类型置信度 < 0.5 | 标记为 `unknown`，跳过信息提取 |
| 图片超过 5 MB / 8000 px | 自动 JPEG 压缩 + 分辨率缩放，缓存避免重复压缩 |
| 单张图纸处理失败 | 记录错误日志，继续处理下一张 |
| API Key 未配置 | 启动时立即报错（fail-fast），不进入处理循环 |

---

## 性能基准

| 场景 | 参考耗时 |
|------|----------|
| 单张图纸（交叉验证一致）| 约 4 至 6 分钟 |
| 单张图纸（触发多轮交叉验证）| 约 8 至 15 分钟 |
| PDF 转图（600 DPI，per page）| 约 3 至 5 秒 |
| OCR 识别（per image）| 约 5 至 30 秒 |

> **重要**：所有 Bash 工具调用 `timeout` 必须设置为 `1200000`（20 分钟），详见 `CLAUDE.md`。

---

## 已知限制

| 任务 | 可靠性 | 说明 |
|------|--------|------|
| 图纸类型识别（平面 / 立面）| 高 | 语义判断，准确率稳定 |
| 文字标注、轴线编号读取 | 高 | OCR + LLM 双重验证 |
| 识别结构体系 | 中 | 宏观判断，不涉及精确计数 |
| 精确统计构件数量 | 低 | Transformer 不擅长系统性枚举，密集符号易漏数 |
| 构件精确坐标（mm 级）| 低 | LLM 输出坐标为语义估计，非测量值 |

**适合场景：** 快速了解图纸类型、提取标高/轴网文字标注、辅助人工审核。
**不适合场景：** 生产级 BIM 精确数据、结构安全验算。

---

## 资源索引

| 文件 | 说明 |
|------|------|
| `scripts/process_drawings.py` | 主入口，支持 CLI 全部参数 |
| `scripts/enhanced_image_processor.py` | 增强处理器（交叉验证 + 上下文管理）|
| `scripts/image_processor.py` | 基础处理器（Prompt 构建、JSON 解析）|
| `scripts/cross_validation.py` | 双模型交叉验证逻辑 |
| `scripts/context_manager.py` | 多轮对话上下文管理 |
| `scripts/client_interfaces.py` | OCR / LLM 客户端接口（含大图压缩、500 自动恢复）|
| `scripts/pdf_to_image.py` | PDF -> 图片转换（支持任意 DPI）|
| `scripts/image_preprocessor.py` | 图像预处理（去噪、对比度增强）|
| `scripts/json_to_vtu.py` | JSON -> VTU 三维重建（需 pyvista）|
| `scripts/config_validator.py` | 配置文件校验 |
| `scripts/logger.py` | 日志模块 |
| `ocr_service/ocr_server.py` | PaddleOCR HTTP 服务 |
| `ocr_service/ocr_cli.py` | OCR 命令行测试工具 |
| `config.json` | 主配置文件（含 API Key 占位字段）|
| `CLAUDE.md` | Claude Code 执行规范（超时、OCR 管理）|
