# Qwen RLCD 本地决策测试

基于 Apple Silicon / MLX 的结构化决策与候选项概率测试，包含本地离线启动脚本和中文 Python 示例。

## Fork 来源

本项目 fork / 派生自 **[harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)**，原作者为 **Harsha Gundala**。

- 上游托管于 Hugging Face；本仓库是其代码在 GitHub 上的派生版本，并非 GitHub 原生 fork 关系。
- 基于上游提交 [`2af86848be75847ccb3553b0941cc51d6ef7e4e9`](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/commit/2af86848be75847ccb3553b0941cc51d6ef7e4e9)，保留原始 Git 历史及作者信息。
- 上游声明许可证为 **Apache-2.0**；原始说明保存在 [README.upstream.md](README.upstream.md)。
- 实际模型为 [mlx-community/Qwen2.5-1.5B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit)，权重不包含在本仓库。固定版本记录在 [model-manifest.json](model-manifest.json)。

## 核心原理

阅读 [run_parallel_generation 原理与实现](README.run_parallel_generation.zh-CN.md)：提示词构造、KV 缓存、字段并行评分、候选概率、冲突分支、TTFT 与扩展方法。内容按当前源码说明，并标注实现限制。

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

## Snake Lab：模型控制贪吃蛇

```bash
./start.sh
```

打开 **http://127.0.0.1:8000/snake.html**。原来的结构化决策页面保留在 `/`。

- **模型驾驶**：每步调用本地 Qwen，比较直行、左转、右转的 A/B/C 分数。默认采用规划辅助，先筛选可行短路径，再由模型在剩余候选中选择。
- **手动操作**：方向键 / WASD 或页面方向按钮；空格暂停，R 重开。
- **实时观察**：显示三个候选动作的相对概率、推理耗时、最近 12 步及原始观测文本。
- **防撞保护**：过滤下一步撞墙/撞身体的动作，介入单独计数；原版和实验版可关闭。规划辅助默认启用防撞，并另行记录规划对动作的调整。
- **速度**：可调整步进间隔；游戏等待每次推理完成，不并发堆积请求。

### 三种策略

- **规划辅助（默认）**：模拟移动的蛇身、尾巴腾空和进食增长；用 6 步生存预测、进食后逃生估计、当前食物的可行路线和重复访问记录筛选动作，模型在候选中选择。可能由规划规则改变模型首选，面板明确标记原因，模型概率不会被改写。
- **丰富状态（实验对照）**：向模型增加上述预测、最近 24 步访问统计、未进食步数；不做规划筛选。实测小模型仍可能循环，不建议用于追求分数。
- **原始模型（对照）**：保留优化前的状态与提示词。

记忆在重开和吃到新食物时清空，避免上一个食物目标的路径干扰当前目标。输入历史最多 24 项。策略切换会丢弃旧推理响应，下一步使用新策略。

预测不会假设下一颗随机食物的位置。先寻找并逐步验证静态候选路径，再用最多 40 层、每层 24 个蛇身状态的束搜索补充；搜索未找到路线只表示预算内未找到。尾巴连通性是静态估计，6 步预测也不保证长期存活，因此规划辅助仍可能失败。

接口为 `POST /api/snake/decide`，请求示例：

```json
{"size":16,"snake":[[8,8],[7,8],[6,8]],"food":[11,8],"direction":1,"safety":true,"policy":"hybrid","recent_heads":[],"steps_since_food":0}
```

坐标 `[x,y]`，左上角为原点，y 向下；方向 0/1/2/3 分别为上/右/下/左。响应包含 `action`、`model_action`、`candidates`、`safety_override`、`planning_override`、`assistance_reason`、`elapsed_ms`（模型耗时）、`total_ms`（总决策耗时）和 `observation`。API 未传 policy 时默认 enhanced，网页显式使用 hybrid。概率为保护筛选前的原始候选分数。三个方向都不安全且保护开启时，`action` 为 null，`trapped` 为 true。

主要文件：`server/snake.py`（观测/模型策略/API）、`server/snake_planning.py`（动态预测与辅助筛选）、`web/snake-engine.mjs`（游戏规则）、`web/snake.mjs`（界面和控制循环）、`web/snake.html` / `web/snake.css`（页面）。修改 `server/snake.py` 中的决策描述即可实验不同策略。

测试：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/snake-engine.test.mjs
```

已通过 15 项后端测试与 6 项游戏规则测试，包括移动尾巴、进食增长、延迟陷阱、记忆重置和输入限制。两个固定种子、各 80 步的本机对比：原版共吃到 8 个食物，规划辅助共吃到 17 个；丰富状态实验版为 0。完整结果及限制见 [优化评估](docs/snake-evaluation.md)。

复现（会执行真实本地推理，耗时数分钟）：

```bash
.venv/bin/python tests/benchmark_snake.py --steps 80 --seeds 2
```

默认报告写入被 Git 忽略的 `work/snake-benchmark.json`。

## 验证情况

已在 Apple M4 上实际运行 `decision_demo.py`，检查候选项数量和概率和；模型文件尺寸、权重 SHA-256、Safetensors 索引、本地分词器和依赖检查均通过。原始结构化比较网页的端到端测试未执行；新增 Snake 页面已实际验证。上游性能及校准声明见原始 README，不代表本仓库独立验证的结果。

详细使用说明见 [README.local.zh-CN.md](README.local.zh-CN.md)。
