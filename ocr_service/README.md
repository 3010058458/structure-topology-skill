# PaddleOCR 服务

基于 PaddleOCR 的 OCR 识别服务，支持文字识别、坐标定位和置信度可视化。

## 功能特性

- ✅ 高精度中文 OCR 识别
- ✅ 自动文档方向检测和矫正
- ✅ 返回文字、坐标和置信度
- ✅ 置信度可视化（彩色边框标注）
- ✅ 支持中文路径
- ✅ 自动重试机制，提高稳定性

## 快速开始

### 1. 安装依赖

```bash
pip install paddlepaddle paddleocr flask requests opencv-python numpy pillow
```

### 2. 启动服务

```bash
python ocr_server.py
```

服务将在 `http://localhost:5000` 启动。

### 3. 使用命令行工具

**仅识别文字：**
```bash
python ocr_cli.py image.jpg
```

**显示完整信息（文字、坐标、置信度）：**
```bash
python ocr_cli.py image.jpg --mode full
```

**绘制置信度框并保存：**
```bash
python ocr_cli.py image.jpg --draw-boxes output.png
```

## 命令行参数

```
python ocr_cli.py <图片路径> [选项]

选项:
  -m, --mode {full,text}    识别模式 (默认: text)
                            full: 显示文字、坐标、置信度
                            text: 仅显示文字

  -d, --draw-boxes OUTPUT   绘制置信度框并保存到指定路径
                            蓝色框: 置信度 ≥ 90%
                            紫色框: 置信度 75-90%
                            红色框: 置信度 < 75%

  -s, --server URL          OCR 服务器地址
                            (默认: http://localhost:5000)

  -v, --verbose             显示详细信息
```

## API 接口

### 1. 健康检查

```bash
GET /health
```

**响应：**
```json
{
  "status": "ok",
  "message": "OCR service is running"
}
```

### 2. 完整 OCR 识别

```bash
POST /ocr
Content-Type: multipart/form-data

file: <图片文件>
```

**响应：**
```json
{
  "success": true,
  "count": 10,
  "result": [
    {
      "text": "识别的文字",
      "confidence": 0.9856,
      "box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    }
  ],
  "rotated_image": "temp_image_xxx_rotated.png",
  "message": "图片已旋转，坐标基于旋转后的图片"
}
```

### 3. 仅文字识别

```bash
POST /ocr/text-only
Content-Type: multipart/form-data

file: <图片文件>
```

**响应：**
```json
{
  "success": true,
  "text": "识别的文字内容\\n多行文字用换行符分隔"
}
```

## 重要说明

### 图片旋转处理

- 服务会自动检测图片方向并旋转为正向
- 如果图片被旋转，会保存旋转后的图片（如 `temp_image_xxx_rotated.png`）
- **返回的坐标是基于旋转后的图片**，确保坐标准确
- 使用 `--draw-boxes` 时会自动在旋转后的图片上标注

### 置信度可视化

使用 `--draw-boxes` 参数可以生成带有彩色边框的标注图片：

- **蓝色框**：高置信度（≥ 90%）
- **紫色框**：中置信度（75-90%）
- **红色框**：低置信度（< 75%）

低置信度和中置信度的框会显示具体的置信度百分比。

### 坐标格式

每个文本框返回4个点的坐标：
```python
[[x1, y1],  # 点1
 [x2, y2],  # 点2
 [x3, y3],  # 点3
 [x4, y4]]  # 点4
```

坐标按顺序连接形成文本框的边界。

## 使用示例

### 示例 1：识别图片并显示文字

```bash
python ocr_cli.py document.jpg
```

输出：
```
这是识别的文字内容
第二行文字
第三行文字
```

### 示例 2：显示完整信息

```bash
python ocr_cli.py document.jpg --mode full
```

输出：
```
识别到 3 行文字:

[1] 这是识别的文字内容
    置信度: 0.9856 (98.56%)
    位置: 左上(100,50) 右下(300,80)

[2] 第二行文字
    置信度: 0.9723 (97.23%)
    位置: 左上(100,90) 右下(250,120)
...
```

### 示例 3：生成置信度可视化图片

```bash
python ocr_cli.py document.jpg --draw-boxes result.png
```

输出：
```
识别到 187 个文本框

已保存标注图片到: result.png

统计信息:
  总计: 187 个文本框
  高置信度 (≥90%): 165 个 (蓝色)
  中置信度 (75-90%): 18 个 (紫色)
  低置信度 (<75%): 4 个 (红色)
```

### 示例 4：使用远程服务器

```bash
python ocr_cli.py image.jpg --server http://192.168.1.100:5000
```

## 文件说明

```
.
├── ocr_server.py       # OCR 服务器（Flask）
├── ocr_cli.py          # 命令行工具
└── README.md           # 本文档
```

## 技术细节

### 模型配置

服务使用以下 PaddleOCR 配置：
- 语言：中文（`lang='ch'`）
- 文本行方向检测：启用
- 文档方向分类：启用
- 文档展平：启用

### 性能优化

- 使用递归锁（RLock）保护 OCR 实例
- 自动重试机制（最多2次）
- 连续错误时自动重新加载模型
- 多线程支持（Flask threaded=True）

### 错误处理

- 自动检测并处理文件路径问题
- 支持中文路径（使用 cv2.imdecode）
- 详细的错误信息和堆栈跟踪
- 超时保护（默认1200秒）

## 常见问题

### Q: 为什么坐标和原图对不上？

A: 如果图片被自动旋转了，返回的坐标是基于旋转后的图片。请使用返回结果中的 `rotated_image` 字段获取旋转后的图片路径。

### Q: 如何提高识别准确率？

A:
1. 确保图片清晰，分辨率足够
2. 避免图片倾斜或扭曲
3. 确保文字对比度足够（深色文字，浅色背景）

### Q: 服务启动很慢？

A: 首次启动时需要下载和加载模型，这是正常现象。后续启动会快很多。

### Q: 支持哪些图片格式？

A: 支持常见的图片格式：JPG, PNG, BMP, TIFF 等。

## 许可证

本项目基于 PaddleOCR 开发，遵循 Apache 2.0 许可证。

## 致谢

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 百度飞桨 OCR 工具库
