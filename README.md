# Qwen RLCD 本地决策测试

基于 Apple Silicon / MLX 的结构化决策与候选项概率测试，包含本地离线启动脚本和中文 Python 示例。

## Fork 来源

本项目 fork / 派生自 **[harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)**，原作者为 **Harsha Gundala**。

- 上游托管于 Hugging Face；本仓库是其代码在 GitHub 上的派生版本，并非 GitHub 原生 fork 关系。
- 基于上游提交 [`2af86848be75847ccb3553b0941cc51d6ef7e4e9`](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/commit/2af86848be75847ccb3553b0941cc51d6ef7e4e9)，保留原始 Git 历史及作者信息。
- 上游声明许可证为 **Apache-2.0**；原始说明保存在 [README.upstream.md](README.upstream.md)。
- 实际模型为 [mlx-community/Qwen2.5-1.5B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit)，权重不包含在本仓库。固定版本记录在 [model-manifest.json](model-manifest.json)。

## 本仓库的修改

- `start.sh`：使用项目虚拟环境和本地权重离线启动网页，仅监听本机。
- `download-model.sh`：下载指定版本的模型。
- `decision_demo.py`：自定义问题和候选决策，返回候选项概率；使用单 token 标签并检测冲突。
- `smoke-test.py`：真实推理和输出结构检查。
- `requirements-lock.txt`：记录安装的依赖版本。
- `core/engine_mlx.py`：支持通过 `MODEL_ID` 环境变量指定本地模型。

## 安装

需要 Apple Silicon Mac、支持 MLX 的 macOS 和 Python 3.12。本机验证环境为 macOS 26.2、Apple M4。

```bash
# 克隆本仓库后进入项目目录
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
./download-model.sh
```

模型下载约 0.87 GB。虚拟环境、权重和缓存均被 Git 忽略。下载需要能连接 Hugging Face；可按自己的网络环境设置 `HTTPS_PROXY`。

## 测试决策概率

```bash
.venv/bin/python decision_demo.py "用户说：我的软件用不了。应该采取什么下一步行动？"
```

修改 `decision_demo.py` 顶部的 `OPTIONS`，保留 `A/B/C` 等单 token 标签，修改标签对应的决策描述。支持 2–5 个候选项。

也可以在项目目录中的 Python 脚本内调用：

```python
from decision_demo import decide

result = decide("用户说：我的软件用不了。应该采取什么下一步行动？")
print(result["candidate_probabilities"])
```

同一进程内模型只加载一次，适合循环测试多条输入。

**概率解释：** 这些是给定候选集合内归一化的模型分数，并非经过真实数据校准的正确率。候选项、提示词和温度都会影响数值。上游遇到部分多 token 冲突时会采用启发式处理，将胜出项概率下限设为 0.75；此示例通过单 token 和冲突检查避开该分支。原始网页仍使用上游逻辑。

## 启动网页

```bash
./start.sh
# 如果端口被占用：PORT=8001 ./start.sh
```

打开 http://127.0.0.1:8000 ，按 Ctrl+C 停止。下载完成后启动和推理均可离线进行。

## 验证情况

已在 Apple M4 上实际运行 `decision_demo.py`，检查候选项数量和概率和；模型文件尺寸、权重 SHA-256、Safetensors 索引、本地分词器和依赖检查均通过。网页端到端测试未执行。上游性能及校准声明见原始 README，不代表本仓库独立验证的结果。

详细使用说明见 [README.local.zh-CN.md](README.local.zh-CN.md)。
