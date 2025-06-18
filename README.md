<div align="center">
  <h1>香港安老院爬虫</h1>
  
  ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Gemini API](https://img.shields.io/badge/Gemini_API-Integrated-green)
</div>

## 项目简介
本项目旨在爬取香港安老院的详细信息，并利用 Gemini API 对爬取到的地址进行行政区识别，以丰富数据维度。

## 特性
- 自动识别安老院地址中的香港行政区（例如：九龙, 深水埗区）。
- **API Key 有效性测试**：程序启动时会自动测试所有提供的 API Key 的有效性，确保只使用可用的 Key。
- 支持通过命令行参数自定义 Gemini 模型和提示词。
- **支持多密钥轮转**：当一个 API 密钥达到速率限制或出现问题时，程序会自动切换到下一个可用的密钥。当检测到返回 429 错误时，会立即转用下一个 key，无需等待时间。当检测到轮转的 key 曾经使用过时，根据该 key 上一次调用时间与当前时间进行判断，如果超过 60 秒则不用等待，直接使用，如果没有超过 60 秒，则等待距离 60 秒的差值时间，提高爬取稳定性。

## 环境配置
在运行本项目之前，请确保您的系统满足以下环境要求并安装必要的依赖。

### Python 版本
- Python 3.10 或更高版本

### 安装依赖
通过 `pip` 命令安装项目所需的 Python 库：
```bash
pip install -r requirements.txt
```

## 配置说明
为了方便配置，本项目提供了一个 `.env.example` 文件作为模板。请按照以下步骤进行配置：

1.  **复制模板文件**：将项目根目录下的 `.env.example` 文件复制一份，并重命名为 `.env`。
    ```bash
    cp .env.example .env
    ```
2.  **编辑 `.env` 文件**：打开新创建的 `.env` 文件，根据您的需求修改其中的配置项。
    -   `GEMINI_API_KEYS`: 您的 Gemini API 密钥。如果需要使用多个密钥，请用逗号分隔。
    -   `BASE_URL`: 爬虫的目标网址。
    -   `OUTPUT_FILE`: 输出 Excel 文件的名称。
    -   `OVERWRITE`: 是否覆盖已存在的输出文件 (`True` 或 `False`)。
    -   `GEMINI_MODEL`: 指定用于行政区识别的 Gemini 模型。
    -   `PROMPT`: 自定义行政区识别的提示词。

    示例 `.env` 文件内容：
    ```ini
    GEMINI_API_KEYS=your_api_key_here
    BASE_URL=https://elderlyinfo.swd.gov.hk/tc/search-result?n%5B0%5D=25&n%5B1%5D=26&items_per_page=50
    OUTPUT_FILE=護老院-爬虫.xlsx
    OVERWRITE=False
    GEMINI_MODEL=gemini-2.0-flash
    PROMPT="提取以下香港地址的行政区(18区)和街道"
    ```
    **注意：** `.env` 文件不应提交到版本控制系统（如 Git），因为它可能包含敏感信息。

### 设置 Gemini API 密钥
本项目使用 Gemini API 进行地址行政区识别。您可以将 API 密钥设置为系统环境变量，或者在项目根目录下创建 `.env` 文件来配置。

**通过 `.env` 文件配置 (推荐):**
在项目根目录下创建名为 `.env` 的文件，并添加以下内容。如果需要使用多个 Gemini API 密钥，请用逗号分隔：
```
GEMINI_API_KEYS=您的密钥1,您的密钥2,您的密钥3
```
**注意：** `.env` 文件不应提交到版本控制系统（如 Git），因为它可能包含敏感信息。

**通过系统环境变量配置 (备选):**
```bash
export GEMINI_API_KEYS="您的密钥1,您的密钥2" # macOS/Linux
# 或者在 Windows 命令提示符中：
# set GEMINI_API_KEYS=您的密钥1,您的密钥2
# 或者在 PowerShell 中：
# $env:GEMINI_API_KEYS="您的密钥1,您的密钥2"
```
**重要提示：** 程序会在启动时测试所有提供的 `GEMINI_API_KEYS` 的有效性。只有有效的密钥才会被用于行政区识别。如果所有密钥都无效或未设置，行政区识别功能将自动禁用，程序会保留原始地址信息。

## 运行说明
执行以下命令启动爬虫程序：

```bash
python main.py
```

### 命令行参数
程序支持以下命令行参数，以提供更灵活的控制：

- `--base-url`: 爬虫的目标网址。
  - **必需参数**。
  - 示例: `python main.py --base-url "https://example.com/new-url"`
  - **如何获取目标网址**：
    1.  在浏览器中访问香港社会福利署长者信息网的安老院搜索页面：[https://elderlyinfo.swd.gov.hk/tc/ltc_search/rcs](https://elderlyinfo.swd.gov.hk/tc/ltc_search/rcs)
    2.  在该页面上，您可以选择所需的筛选条件（例如地区、安老院种类等）。
    3.  点击“搜寻”按钮，页面将跳转到搜寻结果页面。
    4.  在搜寻结果页面，找到“每页显示”选项，并点击“50”按钮，让每页显示 50 条数据。这有助于减少爬虫的翻页次数，加快数据处理速度。
    5.  此时浏览器地址栏中的 URL 即为您需要的爬虫目标网址，将其复制并作为 `--base-url` 参数的值。

- `--output`: 输出 Excel 文件名。在**追加模式**下（即不使用 `--overwrite` 参数时），程序会自动在文件名后添加时间戳以避免覆盖。
  - 默认值: `護老院-爬虫.xlsx`
  - **优先级**: 命令行参数优先于 `.env` 文件中的 `OUTPUT_FILE` 配置。
  - 示例: `python main.py --output "my_elderly_homes.xlsx"`

- `--overwrite`: 启用覆盖模式。如果指定，程序将覆盖已存在的输出文件。
  - 默认行为: 追加数据并自动去重（基于“院舍名稱”和“地址”的组合）。
  - **优先级**: 命令行参数优先于 `.env` 文件中的 `OVERWRITE` 配置 (`True` 或 `False`)。
  - 示例: `python main.py --overwrite`

- `--gemini-model`: 指定用于行政区识别的 Gemini 模型。
  - 默认值: `'gemini-2.0-flash'`
  - **优先级**: 命令行参数优先于 `.env` 文件中的 `GEMINI_MODEL` 配置。
  - 示例: `python main.py --gemini-model 'gemini-2.0-flash'`
  - **可用模型参考**: 请查阅 [Google Gemini API 官方文档](https://ai.google.dev/models/gemini) 获取最新和可用的模型列表。

- `--prompt`: 自定义行政区识别的提示词。
  - 默认值: `"请从以下地址中提取香港的行政区和街道信息。请严格按照 '行政区, 街道' 的格式返回，不要包含其他任何文字或解释。"`
  - **优先级**: 命令行参数优先于 `.env` 文件中的 `PROMPT` 配置。
  - 示例: `python main.py --prompt "请从以下地址中提取香港的行政区和街道信息。请严格按照 '行政区, 街道' 的格式返回，不要包含其他任何文字或解释。"`

## 数据保存与输出
程序运行完成后，将生成包含行政区信息的安老院 Excel 数据。具体输出路径和文件名将根据 `--output` 参数确定。在**追加模式**下，文件名会自动添加时间戳。

### 输出数据结构
以下是 Excel 文件中每条记录的数据结构示例：

```json
{
  "院舍名稱": "安老院名称",
  "地址": "完整地址",
  "電話": "联系电话",
  "行政区": "九龙区",
  "街道": "街道信息"
}
```

### 数据保存逻辑
1.  **去重机制**：在爬取数据并保存之前，程序会加载已存在的 Excel 文件（如果存在），并根据“院舍名稱”和“地址”的组合生成唯一标识符（MD5 哈希值）集合。这用于确保最终保存的数据中不会有重复的安老院记录。
2.  **保存模式**：
    *   **覆盖模式 (`--overwrite` 参数或 `.env` 中的 `OVERWRITE=True`)**：程序会将所有新爬取到的数据直接写入指定的 Excel 文件，完全覆盖文件中的原有内容。
    *   **追加模式 (默认行为，即不使用 `--overwrite` 或 `.env` 中的 `OVERWRITE=False`)**：
        *   程序会读取现有 Excel 文件中的数据。
        *   将新爬取到的数据与现有数据合并。
        *   对合并后的数据进行去重，去重规则是基于“院舍名稱”和“地址”的组合。如果新数据与旧数据存在重复，则保留新数据（即最新爬取到的信息）。
        *   最终，程序会将所有不重复的数据（包括原有的不重复数据和新增加的不重复数据）追加到 Excel 文件中。
        *   **文件名时间戳**：在此模式下，程序会自动在输出文件名中添加当前时间戳（例如：`護老院-爬虫-YYYYMMDD-HHMMSS.xlsx`），以避免覆盖之前的数据。

## 错误处理
- 当 Gemini API 调用失败或返回无效结果时，程序将保留安老院的原始地址信息，不会中断爬取流程，并会在日志中记录详细错误信息。
- **智能重试机制**：针对 API 调用失败，程序会进行多次重试，并在重试时尝试轮转 API 密钥，以应对临时性问题或速率限制。特别是当遇到 429 速率限制错误时，会立即切换密钥并重试。

## 依赖
本项目依赖以下 Python 库：
- `google-generativeai`
- `selenium`
- `pandas`
- `openpyxl`
- `python-dotenv`
- `certifi`
- `numpy`
- `python-dateutil`
- `pytz`
- `tzdata`

## 贡献
欢迎对本项目进行贡献！如果您有任何改进建议、新功能需求或 Bug 报告，请随时提交 Pull Request 或 Issue。

## 许可证
本项目采用 [MIT 许可证](LICENSE) 和 [MIT 中文版许可证](LICENSE-zh)。

## 作者
- 作者：Mison
- 邮箱：1360962086@qq.com