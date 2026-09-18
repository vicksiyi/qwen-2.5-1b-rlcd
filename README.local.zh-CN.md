# 本地使用

以下命令在项目根目录执行。

## 启动网页

在 macOS 普通终端运行：

```bash
./start.sh
```

打开 http://127.0.0.1:8000 。使用页面内的预设或输入上下文及字段定义，即可比较结构化并行解码和自回归输出。按 Ctrl+C 停止服务。端口冲突时使用 `PORT=8001 ./start.sh`。

启动脚本使用项目内 `.venv` 和本地模型，默认离线运行，仅监听本机。首次加载会预热 Metal 着色器。无需激活环境，无需再下载模型。

## 模型和源码

源码：https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD

该仓库提供解码程序，没有名为 RLCD 的独立权重。按其 MLX 实现，实际加载的是 `mlx-community/Qwen2.5-1.5B-Instruct-4bit`，保存到 `models/qwen2.5-1.5b-instruct-4bit`。主要用于布尔值和枚举字段的结构化分类，并非普通聊天网页。

原始代码保留，仅给 `core/engine_mlx.py` 增加 `MODEL_ID` 环境变量覆盖，便于直接加载本地目录。原始 `run.sh` 未改；请使用新增的 `start.sh`。

## 校验

```bash
.venv/bin/python smoke-test.py
```

该命令执行真实 Metal 推理并检查 JSON 和 schema。首次安装时的 Codex 沙箱无法访问 Metal GPU。后续权限环境变化后，已在本机 Apple M4 上用 `decision_demo.py` 完成真实离线推理，并检查候选项数量及概率和。网页端到端验证尚未执行。

## 环境与重新下载

Python 3.12 虚拟环境 `.venv`；依赖精确版本保存在 `requirements-lock.txt`。请在自己的机器上创建虚拟环境，不要复制他人的 `.venv`。

重新安装依赖：

```bash
.venv/bin/python -m pip install -r requirements-lock.txt
```

需要重新下载模型时运行 `./download-model.sh`。如需代理，按自己的代理地址配置，例如：

```bash
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 ./download-model.sh
```

代理仅用于下载；离线启动不需要代理。

模型下载完成，所有文件尺寸与 Hugging Face 元数据一致；权重 SHA-256、Safetensors 索引和本地分词器加载均通过校验。模型版本记录在 `model-manifest.json`。

## 自定义决策概率测试

```bash
.venv/bin/python decision_demo.py "用户说：我的软件用不了。应该采取什么下一步行动？"
```

修改 `decision_demo.py` 顶部的 `OPTIONS` 自定义候选决策，保留 A/B/C 等单 token 标签；修改 `QUESTION` 设置默认问题。也可以在同目录脚本中 `from decision_demo import decide`，再调用 `decide("你的问题")`。函数返回 `candidate_probabilities`（全部候选项，本示例限 2–5 项）。模型只在首次调用时加载，适合在一个进程里循环测试。

这些是候选项内归一化的模型分数，并非经过实际数据校准的正确率。默认 temperature=1.0；更低温度使分布更集中，不能视为更准确。

上游 `core/engine_mlx.py` 对发生 token 冲突的多 token 候选项采用启发式概率处理，包括将胜出项概率下限设为 0.75，并均分其他项概率。本示例检查单 token 和冲突以避免进入这条分支；原始网页接口未作修改。

一次本机实测（上述默认问题）：B 转人工 0.5955、C 补充信息 0.3187、A 操作步骤 0.0858，推理约 199 ms（不含加载和预热）。这个回答未必是理想决策，也说明高分不能直接当作正确率。
