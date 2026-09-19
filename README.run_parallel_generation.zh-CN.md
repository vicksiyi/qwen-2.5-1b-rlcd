# `run_parallel_generation` 原理与实现

本文以本仓库的 [core/engine_mlx.py](core/engine_mlx.py) 和 [core/schema.py](core/schema.py) 为准，解释现有实现，而非描述一个理想化算法。

`run_parallel_generation(context, schema, temperature=1.0)` 将 Qwen 的“预测下一个 token”能力用于有限选项决策：先计算公共上下文，再批量询问每个字段，提取候选 token 的分数，最后由 Python 组装结果。

它是 **Qwen 语言模型之上的推理与评分逻辑**。当前本地启动脚本加载 `Qwen2.5-1.5B-Instruct` 的 MLX 4bit 权重；该函数没有修改模型结构、增加分类头或执行强化学习训练。项目名称中的 RLCD 不意味着每次调用都会训练模型。兼容名称 `run_rlcd_generation` 直接指向同一个函数。

上游代码来源见 [项目 README](README.md#fork-来源)。本文说明的优化主要发生在提示词组织、KV 缓存复用、批处理和输出组装层。

## 1. 输入与输出

| 参数 | 用途 |
| --- | --- |
| `context` | 当前问题、环境状态及需要模型判断的信息 |
| `schema` | `StructuredSchema` 实例，定义字段、类型、候选值及说明 |
| `temperature` | 候选 logits 的缩放温度，默认 1.0 |

目前支持 `boolean` 以及 `enum`（别名 `choice`、`selection`）；枚举最多 255 个候选。不支持直接在 schema 中定义自由文本、任意浮点数或嵌套对象。schema 应非空，枚举候选应使用不重复的字符串。

一个字段可表示下一步操作，多个字段可表示不同维度的分类。例如：

```python
schema = StructuredSchema({
    'action': {
        'type': 'enum',
        'choices': ['A', 'B', 'C'],
        'description': '选择下一步：A=继续处理；B=询问用户；C=转人工。',
    },
    'urgent': {
        'type': 'boolean',
        'description': '是否需要立即处理；true=立即处理，false=不紧急。',
    },
})
```

输出包括 `parsed_json` 和 `field_telemetry`。`parsed_json` 中每个字段是 `{"value": ..., "prob": ...}`，不是直接输出原始 schema 对应的平面 JSON。`field_telemetry` 包含类型、候选数量、首选分数和最多五个候选分数。

## 2. 整体流程

```text
context + schema
       │
       ├─ 编译每个字段的后缀、候选 token 和冲突标记
       │
       └─ 构造公共提示词
                 ↓
         公共前缀 prefill，一次模型调用
                 ↓
         得到每层的 KV cache
                 ↓
         沿 batch 维复制成 M 份
                 ↓
         M 个字段后缀，一次批量模型调用
                 ↓
         读取每行最后一个有效位置的 logits
                 ↓
         截取候选 token 分数 → softmax → argmax
                 ↓
         Python 组装结构化结果和统计信息
```

这里 `M` 是字段数量。不是为每个候选运行一遍完整模型，也不是启动 M 个独立模型进程。若仅有一个 `action` 字段，字段 batch 大小就是 1；候选 A/B/C 的分数来自该位置的同一个词表 logits 向量。

正常无冲突路径共有 **两次模型调用：prefill + batched suffix**。所谓“一次并行”指后缀阶段的一次批量调用，不是从原始输入到结果只做一次模型调用。

## 3. schema 如何编译

`StructuredSchema.compile_parallel_metadata(tokenizer)` 为各字段准备以下数据：

- `field_items`：字段名称与定义。
- `suffix_lengths`：各字段后缀实际 token 长度。
- `cands_per_field`：每个候选对应的一个决策 token ID。
- `prefixes`：同字段所有枚举字符串的公共字符前缀。
- `has_collisions`：同字段是否有候选映射到同一个 token ID。
- `suffixes_batch`：右侧补齐到相同长度的后缀 token 矩阵。

对于布尔字段，后缀类似 `  "urgent": `，候选取 `true` 和 `false` 各自编码的第一个 token。

对于枚举字段，先用 `os.path.commonprefix()` 提取公共**字符**前缀。候选为 A/B/C 时公共前缀为空，后缀类似 `  "action": "`，候选 token 来自 A、B、C。若候选都以 `status_` 开头，该前缀会写入字段后缀，再对剩余部分编码取第一个 token；实际切分必须以 tokenizer 结果为准。

编译结果缓存于 schema 实例。它不是全局缓存：每次新建 schema 都会重新编译。缓存也没有按 tokenizer 或 schema 修改自动失效，因此更换模型、修改候选时应重新创建 schema。

注意，`compile_candidate_tokens()` 和 `extract_calibrated_probabilities()` 虽然也在同一模块中，但不是这个函数实际使用的主评分路径。不要将它们的多变体 token 汇总逻辑误认为这里已经采用。

## 4. 公共提示词如何构造

函数手动拼接 Qwen 风格的 ChatML：

```text
<|im_start|>system
Classify JSON attributes:
  "action": 选择下一步：A=继续处理；B=询问用户；C=转人工。
  "urgent": 是否需要立即处理；true=立即处理，false=不紧急。<|im_end|>
<|im_start|>user
这里是 context 的内容<|im_end|>
<|im_start|>assistant
{
```

一个容易忽略的细节：`to_parallel_schema_str()` 只放入字段名和 **description 的第一行**，不会自动把 `choices` 列表完整写进提示词。因此使用 A/B/C 标签时，必须在 description 第一行解释每个标签的含义。

字段 schema 既用于向模型说明任务，也用于程序截取候选分数。仅在 `choices` 中列出 A/B/C，却不解释含义，模型无法据此理解你的业务决策。

这里没有调用 tokenizer 的聊天模板接口；更换为其他模型家族时，应检查模板、特殊 token、分词和缓存实现，而不是只改模型路径。

## 5. KV cache 为什么能节省计算

Transformer 在处理序列时，会为各层保存已经计算的 Key 和 Value。后续 token 可以使用这些缓存，无需重新为前文计算相同的 K/V。

该函数先对公共提示词做一次 prefill，再用 `copy.copy()` 复制各层缓存对象，并通过 `mx.repeat(..., M, axis=0)` 沿 batch 维复制 K/V 张量。于是每行字段后缀都可以从相同上下文继续计算。

后缀矩阵形状为 `[M, S_max]`。模型输出 logits 可理解为 `[M, S_max, V]`，其中 V 为词表大小。字段 i 使用：

```python
field_logits = suffix_out[i, suffix_lengths[i] - 1, :]
```

右侧 padding 不影响它之前的有效位置的因果预测；函数读取的是最后一个真实后缀 token 对应的位置，而不是整行最后一个补齐位置。

这会减少重复的前缀计算，但不是零成本共享：`mx.repeat` 带来随字段数增长的缓存存储开销，较长上下文与较多字段仍可能增加显存/统一内存占用和延迟。每次函数调用都会新建 prompt cache，当前没有跨请求复用前缀缓存。

多个字段各自从同一个公共上下文分支预测，彼此看不到其他字段的最终答案。因此它们可能出现逻辑冲突；这个 batch 不是对多个字段联合结果的严格概率建模。

## 6. 候选分数如何变成“概率”

无 token 冲突时，取候选 token 的 logits，除以温度，再只在候选集合内计算 softmax：

```text
p(j) = exp(z(j) / T) / Σ exp(z(k) / T)
                           k 属于候选集合
```

假设 A/B/C 的 logits 为 `[2, 1, 0]`，T=1 时，候选内分数约为 `[0.6652, 0.2447, 0.0900]`。这是公式示例，不是本机模型的一次实测。

函数用 `argmax` 选最大项，没有按概率随机采样。正温度主要改变分布的尖锐程度，理论上不改变这组固定 logits 的大小排序。代码用 `max(temperature, 1e-4)` 做下限处理；调用层仍应校验输入是有限正数，可参考 `decision_demo.py`。

这些值表示**当前提示词与候选集合内的相对模型偏好**：

- 候选集合改变，归一化分数也会改变。
- 候选以外的全部词表概率质量被忽略，所以较高分数不证明模型原本强烈倾向该动作。
- 它没有经过标注集上的正确率校准；字段名中的 `confidence`、`calibrated` 不能作为统计校准的证据。
- 非冲突也不代表完整字符串已评分：多 token 候选即使首 token 不冲突，正常路径仍只使用剩余部分的首 token，没有计算完整候选序列似然。

输出四舍五入到四位小数。`top_choices` 最多保留五项，候选超过五个时，返回的五项概率之和可能小于 1。

## 7. 多 token 冲突分支的实际行为

如果多个候选映射到同一个决策 token，函数进入启发式消歧：

1. 从批量缓存中切出该字段的一行。
2. 按全词表 `argmax` 最多尝试四轮续写，遇到引号、换行或逗号终止。
3. 将续写文本与候选做前缀匹配；匹配不到再尝试用数字作为索引，最终仍失败则选择第一个候选。
4. 将所取 token 的概率乘积限制到 `[0.75, 0.9999]`，作为胜出项分数；其余质量平均分配给剩余候选。

**这不是完整候选的严格约束解码，也不是可信的概率校准。** 特别是 0.75 下限是程序设置的，并不代表模型至少有 75% 把握。该分支还可能增加多次模型调用。

另一个需要审计的实现细节：切出的缓存已经处理了补齐后的完整后缀，而首个 `field_logits` 取自真实后缀末端。不同后缀长度下，续写缓存位置可能已经越过 padding，不能未经验证就将这一路径视为精确候选序列打分。

本仓库的 [decision_demo.py](decision_demo.py) 使用互不冲突的单 token 标签并显式检查，避开这条路径。为中文业务动作评分时，可以让 A/B/C 对应完整中文描述，不必让长中文动作名称直接作为被评分 token。

## 8. 输出统计与 TTFT 如何理解

| 返回字段 | 当前实现的含义或限制 |
| --- | --- |
| `elapsed_ms` | `get_engine()` 返回后开始计时，包含 schema 编译、提示词处理、prefill、缓存复制、后缀计算及结果组装 |
| `prefill_ms` | 公共前缀的模型调用与所指定缓存张量的求值时间 |
| `suffix_eval_ms` | 批量后缀模型调用及输出求值时间，不包含前面的缓存复制 |
| `sequential_forward_passes` | 固定返回 1，不能据此认为实际只有一次模型调用 |
| `total_tokens_generated` | 固定返回 0，表达正常路径没有逐 token 生成完整 JSON；不反映冲突分支的续写 |
| `is_valid_json` / `schema_match` | 固定返回 True，结果主要由程序组装；不是独立验证业务正确性的结果 |
| `has_calibrated_probabilities` | 固定返回 True，不表示已完成统计校准 |

`elapsed_ms` 不包括初次模型加载/预热，也不包括进入函数装饰器时等待 GPU 锁的时间。HTTP 请求的实际耗时还会包括队列、请求处理和网络传输。因此它既不是严格的 TTFT 指标，也不等于所有场景下用户看到的决策总耗时。

可以将此方案理解为“将长 JSON 的逐 token 生成，缩短为公共上下文计算和字段决策位置评分”。延迟是否降低、降低多少，应在同一硬件、相同任务规模及冷/热启动条件下测量，不能从固定返回的步数推导加速倍数。

`@gpu_locked` 使调用串行占用共享模型。这里的并行是单次请求内的字段 batch，不是多个 HTTP 请求同时使用 GPU。

## 9. 可运行的最小调用

先按 [项目安装说明](README.md#安装) 准备环境和本地权重。在项目根目录执行：

```bash
.venv/bin/python - <<'PY'
from decision_demo import decide

result = decide('用户只说“软件用不了”，没有错误信息，下一步怎么办？')
print('选择：', result['decision'])
for item in result['candidate_probabilities']:
    print(item['label'], item['decision'], item['probability'])
print('引擎耗时（毫秒）：', result['elapsed_ms'])
PY
```

`decide()` 是安全使用当前评分路径的参考封装：指定本地权重、检查 2–5 个单 token 标签、在 description 第一行写入映射、检查冲突，然后调用 `run_parallel_generation()`。它不对分数做额外的正确率校准。

若要自行扩展，建议先阅读 `decide()`，再直接使用 `StructuredSchema` 与引擎接口。`MODEL_ID` 应在导入 `core.engine_mlx` 前设置，因为模块导入时就读取该环境变量。

## 10. 如何扩展

| 目标 | 扩展位置与方法 |
| --- | --- |
| 新业务分类或路由 | 修改 context、schema description 与 A/B/C 的业务映射 |
| 多维度分类 | 添加字段，关注缓存开销与字段之间的逻辑一致性 |
| 前后依赖决策 | 分阶段调用，将第一阶段选定的结果加入第二阶段 context |
| 需要解释或自由文本 | 额外调用自回归生成；评分函数本身不生成长解释 |
| 游戏控制 | 由应用提供状态和合法动作，将选定标签映射成动作，再收集新状态 |
| 实际可信度评估 | 收集标注集，评估准确率、可靠性曲线等，再考虑温度拟合或其他校准方法 |
| 严格多 token 候选评分 | 计算完整候选的条件序列似然，或实现 trie 约束解码，替换当前启发式分支 |
| 更换模型 | 验证聊天模板、候选 token、缓存结构及 MLX 兼容性 |

在游戏或其他执行系统中，这个函数只负责“根据输入给候选评分”。环境观测、动作是否合法、执行器、历史记录和下一轮循环由应用层承担。状态不完整或候选设计不合理时，单纯加速评分不会让决策自动变聪明。

## 11. 源码阅读顺序

1. [decision_demo.py](decision_demo.py)：理解业务选项如何映射为单 token 标签。
2. [core/schema.py](core/schema.py)：查看 `to_parallel_schema_str()` 和 `compile_parallel_metadata()`。
3. [core/engine_mlx.py](core/engine_mlx.py)：依次看 `get_engine()`、`gpu_locked()`、`run_parallel_generation()`。
4. [server/snake.py](server/snake.py)：查看状态构造、模型评分与应用层动作选择如何衔接。

本文记录现有实现及边界，没有改变引擎算法或模型权重。
