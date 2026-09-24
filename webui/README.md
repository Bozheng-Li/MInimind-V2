# MiniMind WebUI

本项目的 Web 控制台：**对话推理 + 实验结果可视化 + 训练编排**，一个 FastAPI 后端带三个页面。
后端是 Python，前端是**原生 HTML/CSS/JS —— 无构建、无 npm、无 CDN**（见「离线约束」）。

## 为什么不用 `scripts/web_demo.py`

Streamlit 版用 `AutoModelForCausalLM.from_pretrained` 读 **transformers 格式**的目录，而本项目
新训的模型（`gated` 注意力 + `moe_finegrained` 前馈）是**原生 torch `.pth`**，必须按配置组装。
本 WebUI 照搬 `trainer/eval.py` 的架构感知加载路径（`load_config` → `to_arch_config` →
`build_lm_config` → `arch.build_model` → `torch.load`），因此能加载任意组合的架构。

除了加载方式，还修掉了 Streamlit 版的一个隐藏问题：它用
`TextIteratorStreamer(skip_special_tokens=True)`，而 `<tool_call>` / `<think>` 都是
**added token**，会被一起吃掉 —— 工具调用实际解析不出来。这里自带增量解码器，只丢弃
`<|im_start|>` / `<|im_end|>` 等聊天控制符，保留结构标签。

## 启动

```bash
# 默认加载 configs/sft_moe_full.yaml + test/out/full_sft_moe_full_768_moe.pth，端口 7860
python webui/server.py

# 指定配置 / 权重 / 设备 / 端口
python webui/server.py \
    --config configs/sft_moe_full.yaml \
    --weight test/out/full_sft_moe_full_768_moe.pth \
    --device cuda:0 --port 7860

# 多卡机器上限制可见 GPU（等价于 --device cuda:0）
CUDA_VISIBLE_DEVICES=3 python webui/server.py --config configs/sft_moe.yaml

# 先不加载模型，进界面再选
python webui/server.py --no-load
```

打开 <http://localhost:7860>。启动参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--config` | `configs/sft_moe_full.yaml` | 模型结构配置，**必须与训练时一致**，否则权重形状对不上 |
| `--weight` | 由配置推导 | 权重路径；留空则按 `train.save_dir` + `train.save_weight` + `_<hidden>[_moe].pth` 推导 |
| `--device` | `cuda:0`（无 GPU 则 `cpu`） | 推理设备 |
| `--host` / `--port` | `0.0.0.0` / `7860` | 监听地址与端口 |
| `--no-load` | 关 | 启动时不加载模型 |

### 产物目录 `MINIMIND_ARTIFACT_ROOT`

实验台与训练控制台读的**所有**数据都从这一个根目录出发（默认 `<repo>/test`）：

| 环境变量 | 默认 | 内容 |
| --- | --- | --- |
| `MINIMIND_ARTIFACT_ROOT` | `<repo>/test` | 下面三者的共同父目录 |

```
$MINIMIND_ARTIFACT_ROOT/
├── out/                权重与 *_metrics.csv（训练写）
└── log/
    ├── runs/<run_id>/  每次训练的结构化登记（meta.json + 曲线）
    ├── jobs/<job_id>.{json,log}   WebUI 起过的训练作业
    └── …              训练脚本的原始 .log / .csv / .status
```

`/api/experiments`、`/api/runs`、`/api/logs` 都只看这个根目录；改它就能把界面指向别处的产物。
`test/out/` 下已有的权重会被对话页自动识别（扫描 `configs/*.yaml` 的 `train.save_weight` 与
`test/out/*.pth` 配对，配置是唯一事实来源）：

| 权重 | 配置 | 说明 |
| --- | --- | --- |
| `test/out/pretrain_moe_768_moe.pth` | `configs/pretrain_moe.yaml` | MoE 预训练 base（只会续写，建议关掉 chat 场景使用） |
| `test/out/full_sft_moe_768_moe.pth` | `configs/sft_moe.yaml` | SFT mini（90 万条） |
| `test/out/full_sft_moe_full_768_moe.pth` | `configs/sft_moe_full.yaml` | SFT full（510 万条） |
| `test/out/pretrain_768.pth` | `configs/pretrain.yaml` | 旧的 dense 基线 |

## 三个页面

| 路径 | 页面 | 回答什么问题 |
| --- | --- | --- |
| `/` | **对话** | 这个权重现在能干什么、它是哪次训练出来的 |
| `/lab` | **实验台**（10 页） | 全流程跑到哪了、各算法/架构/评测怎么比 |
| `/train` | **训练控制台** | 下一步该起哪个训练、用什么配置、正在跑的那个现在怎么样了 |

## 指标 → 页面

训练侧采集的每一类指标在界面上都有明确落点。这张表是「想查某项该去哪页」的索引，
也说明了**不同算法记录的日志确实不同**（`trainer/algos/*.py::format_log` 各自格式化）：

| 指标家族 | 出现在 | 在界面上看 |
| --- | --- | --- |
| `loss` / `val_loss` | 全部 | 预训练、监督微调、训练登记 |
| `grad_norm*` | 全部 | 预训练、监督微调（分面）；控制台实时曲线 |
| `tokens/s` / `gpu_mem` / `gpu_util` / `ram` | 全部 | 各页统计牌；控制台实时曲线 |
| `update_ratio_{attn,ffn,embed,norm}` | 预训练、SFT | 预训练「优化动力学」、监督微调三 run 对照 |
| `gate_mean/std/sat_lo/sat_hi` | MoE 架构 | 预训练「门控健康」 |
| `moe_load_e0..e15`（逐专家负载） | MoE 架构 | 预训练「专家负载热力图」 |
| `moe_load_cv` / `moe_entropy_norm` / `dead_experts` | MoE 架构 | 实验库「MoE 健康度横比」；训练登记展开 |
| 逐层 `*_rms_L0..L7` | 预训练、SFT | 预训练「逐层激活量热力图」 |
| 奖励分解 `rew_len/think_len/think_close/rep/rm` | GRPO / DAPO / RLOO / PPO / Agentic | 强化学习「奖励分解」堆叠面积 |
| `eos_rate` / `trunc_rate` / `perplexity` / `ratio_mean` | 在线 RL | 强化学习「生成健康度」 |
| `critic_loss` / `value_loss` / `approx_kl` / `kl_early_stop` / `actor_lr` / `critic_lr` | **仅 PPO** | 强化学习「PPO 专属」卡 |
| `turns_mean` / `tool_calls_mean` / `valid_call_rate` / `tool_gap_mean` | **仅 Agentic** | 强化学习「Agent 专属」卡 |
| `preference_loss` / `reward_margin` / `preference_acc` | DPO / IPO / SimPO / CPO / ORPO / KTO | 强化学习「离线偏好」卡（DPO 另保留 `dpo_loss` 兼容列） |
| 算法 × 指标家族覆盖矩阵 | — | 强化学习页底的覆盖热力图 |
| 标准 benchmark | 评测脚本 | 标准评测；实验库「各阶段最强模型」 |

「出现在」一列指的是指标 CSV 的列清单（`trainer/common/metrics.py` 按算法声明），不是每页都画全 ——
一页只放该页结论需要的图，逐 step 的完整列在 `/lab#logs` 与 `test/out/*_metrics.csv` 里。

### `/` 对话页（三栏推理控制台）

- **模型切换** —— 下拉选择已扫描到的「配置 + 权重」组合，一键加载；也可在「手动指定 config /
  权重」里填任意路径。加载后控制台显示**全槽位架构明细**（norm / positional_encoding /
  attention / feedforward 的类型与关键超参：heads、kv_heads、head_dim、qk_norm、experts 数…）、
  参数量 / 可训练量 / 冻结量（> 0 说明挂了 LoRA）、设备、精度与**权重匹配状态**。
- **训练来源** —— 这个权重是**哪次训练产出的**。模型加载后按**权重文件名**去 `/api/runs`
  （训练侧写的登记表）里精确匹配，命中就展示 `run_id` / 算法 / 起止时间 / 墙钟 / 超参与数据行数 /
  结果摘要（末 loss、最低 loss、最低 val_loss、吞吐、峰值显存、最大梯度范数），并给一个深链到
  `/lab#runs`。同一份权重被训过两次时列出全部（最近一次展开，其余附在末尾）。
  **匹配不上就明说「该权重没有对应的训练登记」**，读取失败则显示失败原因 —— 两者不是一回事，
  界面不猜、也不把「读不到」伪装成「没有」。
- **流式输出** —— SSE 逐 token 推送，生成在独立线程跑、经队列回传，界面有闪烁光标。
- **思考链** —— `open_thinking` 开关；`<think>…</think>` 渲染成可折叠块，流式时显示「思考中」+
  live 标记，结束后显示字数。纯空白的思考块会被自动隐藏。
- **工具调用** —— 内置 8 个工具（数学 / 时间 / 随机 / 字数 / 单位 / 天气 / 汇率 / 翻译），
  最多勾选 4 个。模型输出 `<tool_call>` → 服务端执行 → 结果以 `tool` 角色回灌 → 继续生成，
  **最多 16 轮**。界面上「Tool Calling」蓝色卡片是模型发起的调用，「Tool Called」绿色卡片是
  服务端返回的结果。
- **参数面板** —— `temperature` / `top_p` / `top_k` / `repetition_penalty` / `max_new_tokens` /
  历史轮数 / 随机种子，选择会记在 localStorage。
- **会话管理** —— 新建、清空、导出 Markdown、导出 JSON。
- **性能显示** —— 每条回复下方显示 token 数、tok/s、耗时、工具轮数。
- **停止生成** —— 生成中点方形按钮，通过 `stop_event` 中断阻塞中的 `generate`。
- **运行环境卡** —— 每 5 s 轮询 `/api/resources`，把显存 / 利用率 / 温度 / 功耗画成仪表条，
  并标出「当前有没有卡在跑别人的训练」。
- **界面** —— 深 / 浅主题（记住选择）、响应式（≤1420px 右侧控制台变抽屉、≤1024px 左侧导航变
  抽屉）、Markdown 渲染（标题 / 列表 / 引用 / 表格 / 链接 / 行内代码）、代码块等宽 + 一键复制、
  消息级复制与重新生成。

> SSE `start` 事件里带着「本次推理真正用的是哪份权重」（服务端 `engine.meta.weight`）。
> 「训练来源」卡优先用它 —— 比读已加载模型的记录更贴近事实。

### `/lab` 实验台（10 页）

左侧固定导航，每页只回答一个问题；**刻意混用不同图型**（折线看趋势、横向条看量值、发散条看
「相对基线的增减」、热力图看矩阵、环图看构成、散点看权衡、时间轴看「谁在什么时候跑了多久」、
管线图看流程），不把什么数据都画成折线：

| 页（slug） | 内容 | 数据来源 |
| --- | --- | --- |
| **实验库** `#library` | 产物总览：模型/权重清单、数据集规模、训练次数、评测覆盖、**MoE 健康度横比**、各阶段最强模型 | `test/out/*.pth`、`test/storage/report/**`、`test/log/runs/**` |
| **全流程** `#pipeline` | 从分词器到部署的管线图（节点带真实数字）+ 逐阶段产物清单 | 上面全部来源的汇总 |
| **训练登记** `#runs` | `test/log/runs/` 里的每次训练：架构 / 超参 / 策略 / 资源 / **结果摘要（分组）** | `test/log/runs/<run_id>/meta.json` |
| **预训练** `#pretrain` | loss、吞吐、梯度范数；**优化动力学 / 门控健康 / 逐层激活量热力图 / 专家负载热力图 / 系统** | `test/log/pretrain*.log`（Dense）、`test/out/pretrain_moe_metrics.csv`（MoE，139 列） |
| **监督微调** `#sft` | 各 SFT 实验的 train/val loss、留出集差异、**优化动力学三 run 对照**、逐层 hidden_rms 热力图 | `test/out/sft_{dense,moe,moe_full}_metrics.csv` |
| **强化学习** `#rl` | 奖励、回复长度、通过率；**奖励分解堆叠面积 / 生成健康度 / PPO·Agent·DPO 专属卡 / 算法×指标覆盖矩阵** | `test/out/rl/<algo>_<arch>_metrics.csv` |
| **架构对比** `#sweep` | Dense vs MoE 的分面折线 + 散点（质量 vs 成本） | `test/storage/report/summary.csv`（18 组） |
| **标准评测** `#eval` | 任务热力图（按列归一化配色、读数仍是原值）+ 相对随机的 delta；**任务列由后端动态驱动**，新增任务零前端改动 | `test/storage/report/eval/summary.csv` + `tasks.json` |
| **资源与配置** `#resources` | 实时 GPU / 内存 / CPU / 磁盘 + 生效的配置项 + **起训练前检查表**（数据是否就绪 / 权重产物名 / **实测**峰值显存 vs 当前最大空闲） | `/api/resources`、`/api/catalog`、`/api/models`、`/api/experiments` |
| **日志浏览** `#logs` | `test/log/` 下的原始文件，按需分段读取、高亮 error/warn | `test/log/**` |

> 检查表里的显存列**只报实测值，不做外推**：带 rollout 的 RL 与纯前反向的 SFT 不是同一个
> 基函数（GRPO 的 batch=2 却要同时放 6 条采样序列），用「参数量 → 显存」拟合会解出物理上不
> 成立的系数。没有实测过的配置就如实写「无实测」。

配色遵循 dataviz 规范：分类色固定顺序不循环、顺序色单色相浅→深、发散色两色相 + 中性灰中点、
状态色（good/warning/serious/critical）只表状态且必配图标或文字。**绝不双 Y 轴**；≥2 条序列
必有图例；每张图旁都有可读的表视图兜底 —— 颜色只是加速阅读、不是唯一信息。深浅两套色阶各自
在对应底色上验证过对比度（详见 `static/viz.css` 的注释与实测数字）。

### `/train` 训练控制台

「选算法 → 选配置 → 覆盖超参 → 起进程」，全部走 `trainer/train.py` 的统一入口：

- **表单由后端生成** —— `/api/catalog` 返回可用配置、数据集、算法与策略选项、每个参数的中文
  名/说明/可选值，前端不硬编码任何字段。
- **显示「实际生效」的默认值** —— 选中 (算法, 配置) 后调 `/api/train/defaults`，它在**训练入口
  自己的 argparse 上**解析一遍，因此展示的值和真正跑起来的值同源（三层：argparse 硬编码 →
  `configs/<阶段>.yaml` → `--config` 变体）。
- **命令预览** —— 改任何字段都实时调 `/api/train/preview`，展示**将要执行的确切 argv**。
  预览与执行共用同一个 `build_command()`，两者不可能不一致。
- **实时资源** —— 起训练前先看 GPU 空不空（`/api/resources`）。
- **作业列表** —— 状态、退出码、耗时、实时日志（增量 tail），可停止。
- **实时指标（从日志解析，零服务端改动）** —— 训练 stdout 每行由算法自己格式化
  （`trainer/algos/*.py::format_log`），前端用一个通用的 `键: 值` 解析器把增量日志解析成曲线：
  作业卡里最多 4 张单曲线小图（loss / 奖励 / 梯度范数 / 学习率 / 吞吐 / 显存 / critic_loss /
  approx_kl / DPO 与 Agent 的专属列…按算法实际打出来的取），下面列出**本次算法在 stdout 声明的
  指标 N 个** —— 这正是「不同算法记不同 log」在控制台上的正面体现。
  解析只吃**这次新到的日志分片**（正文的 200 KB 截断不影响解析），指标块**就地替换**而不是整栏
  重绘，所以 2.5 s 的轮询不会重置用户展开的日志与滚动位置。
- **作业 ↔ 训练登记** —— 用 `save_weight` / `run_name` 与 `/api/runs` 的 `run_name` 精确匹配，
  命中才给一条「训练登记：`<run_id>`」链接（对不上就不出声，不猜）。


## 接口

### 推理

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | `{ok, loaded, device}` |
| GET | `/api/models` | 扫描到的模型列表 + 当前已加载模型的信息 |
| GET | `/api/tools` | 8 个工具的定义（OpenAI function-calling 格式） |
| POST | `/api/model/load` | `{config, weight?, device?}`，加载模型 |
| POST | `/api/model/unload` | 卸载并释放显存 |
| POST | `/api/stop` | 中断当前生成 |
| POST | `/api/chat` | 对话；`stream: true` 返回 SSE，`false` 返回聚合 JSON |

`POST /api/chat` 请求体：

```json
{
  "messages": [{"role": "user", "content": "上一轮"}],
  "prompt": "本轮输入",
  "temperature": 0.85, "top_p": 0.85, "top_k": 50,
  "repetition_penalty": 1.05, "max_new_tokens": 1024,
  "history_rounds": 4, "open_thinking": false,
  "tools": ["calculate_math"], "seed": null, "stream": true
}
```

SSE 事件类型：`start` `round_start` `think` `content` `tool_result` `round_end` `done` `error`。

```bash
# 流式
curl -sN -X POST localhost:7860/api/chat -H 'Content-Type: application/json' \
  -d '{"prompt":"现在几点了？","max_new_tokens":200,"tools":["get_current_time"]}'

# 非流式（脚本化验证用）
curl -s -X POST localhost:7860/api/chat -H 'Content-Type: application/json' \
  -d '{"prompt":"计算 (23*7+11)/4","max_new_tokens":200,
       "tools":["calculate_math"],"open_thinking":true,"stream":false}'
```

### 实验数据与日志

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/experiments` | 全部实验数据（预训练 / 监督微调 / 强化学习 / 架构对比 / 标准评测），读盘解析在后台线程 |
| GET | `/api/catalog` | 可用配置、数据集、算法、策略选项、参数字段说明 |
| GET | `/api/resources` | 实时 GPU / 内存 / CPU / 磁盘 / 正在跑的训练进程 |
| GET | `/api/runs` | `test/log/runs/` 的训练登记（对话页的「训练来源」卡读的就是它） |
| GET | `/api/logs` | `test/log/` 下可浏览的原始日志清单 |
| GET | `/api/logs/content` | 读一个日志文件；`path` + `offset` + `limit`（默认 200 kB）分段 |

### 训练编排

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/train/defaults` | `?algo=&config=` → 该组合**实际生效**的默认值 |
| POST | `/api/train/preview` | `{algo, params}` → 将执行的 argv（只拼不跑） |
| POST | `/api/train/start` | `{algo, params}` → 起进程，返回 job meta |
| GET | `/api/train/jobs` | 作业列表（`limit`，默认 50） |
| GET | `/api/train/jobs/{id}` | 单个作业的 meta |
| GET | `/api/train/jobs/{id}/log` | 作业日志增量 tail（`offset` + `limit`） |
| POST | `/api/train/jobs/{id}/stop` | 停止作业（对进程组发信号） |

## 离线约束（重要）

**这个前端没有构建步骤，也不引用任何外部资源。** 所有样式与脚本都是仓库里的文件，字体走系统
字体栈，图标是内联 SVG —— 因此在**完全断网**的机器上打开也与开发机一致。加新依赖前请先确认
它不需要 CDN / npm / bundler。后端的 Python 依赖同样只用到项目已有的（fastapi / uvicorn）。

## 目录

```
webui/
├── server.py           FastAPI 服务端：路由 / SSE 流式 / 工具循环 / 路径防护
├── catalog.py          只读扫描：配置、数据集、算法、资源、字段说明
├── experiments_data.py 读 test/out 与 test/log，解析成实验数据
├── jobs.py             训练作业：拼 argv（白名单）/ 起进程 / 状态 / tail / 停止
├── static/
│   ├── index.html      对话页          ┐
│   ├── lab.html        实验台          ├ 各自引用同一套样式层
│   ├── train.html      训练控制台      ┘
│   ├── style.css       站点令牌 + 基础组件（主题在这儿定义）
│   ├── viz.css         图表令牌（分类/顺序/发散/状态四类色）+ 图元样式
│   ├── lab.css         实验台外壳：卡片 / 徽标 / 键值表 / Toast / 日志视图
│   ├── chat.css        对话页三栏布局 + 响应式抽屉
│   ├── train.css       训练控制台布局
│   ├── viz.js          图表库：折线 / 横向条 / 发散条 / 热力图 / 散点 / 环图 / 管线 / 仪表
│   ├── app.js          对话页：markdown 渲染 + SSE 解析 + 增量 DOM 更新
│   ├── lab.js          实验台 10 页渲染
│   └── train.js        训练控制台：表单 / 预览 / 作业面板
├── screenshots/
└── README.md
```

**样式层叠顺序不可颠倒**：`style.css`（站点令牌）→ `viz.css`（图表令牌）→ 各页外壳
（`lab.css` / `train.css` / `chat.css`）。同名选择器（`.btn` / `.chip` / `.toast` / `.dot` …）
以后加载的为准，所以页面自己的 css 必须放最后。

## 安全边界

WebUI 能起训练进程，因此接口有意收得很紧：

- **参数白名单** —— 训练参数只允许 `jobs.ALLOWED_FLAGS` 里列出的键，其余静默丢弃；键名还要匹配
  `^[a-z][a-z0-9_]{0,63}$`，值里出现 `\x00` / `\n` / `\r` 直接拒绝。
- **不经过 shell** —— 命令是 `argv` 列表，用 `subprocess` 的 `shell=False` 启动，`start_new_session=True`
  单独开进程组（停止时对整组发信号，不会留下孤儿子进程）。
- **config 路径** —— 必须是仓库内的真实文件，越界（`..` / 绝对路径指到仓库外）直接 400。
- **`/api/logs/content`** —— 只允许 `test/log/` 下的白名单后缀，拒绝越界，超过 64 MB 的文件
  拒绝整体读取、只能 `offset`/`limit` 分段。
- **`catalog.py` 只读** —— 它扫描配置和产物用于展示，不修改任何文件、不写配置。
- **`experiments_data.py` 只读数据** —— 它只读 CSV / JSON / 日志，**不 import `test/` 下的任何代码**。
  `#eval` 的任务表也遵守这条：优先读 `report/eval/tasks.json`（由
  `test/storage/eval_suite.py::TASKS` 导出），读不到就退回内置的 `EVAL_TASKS` 常量 ——
  旧产物照样能渲染，边界不变。

## 实现要点

- **生成在线程里跑**：`ArchForCausalLM.generate` 是阻塞的，工作线程把 token 通过
  `loop.call_soon_threadsafe` 投递到 `asyncio.Queue`，异步生成器再转成 SSE。一次只允许一个
  请求占用模型（`Engine.lock`），避免并发抢占显存。
- **增量解码**：单个 token 可能只承载 UTF-8 的一个字节，解码会得到 `�`；这类 token 先
  暂存到 `pending`，等后续字节到齐再一起解码。
- **思考/正文分段**：在增长中的文本上做增量状态机，非结束时扣住末尾 `len(tag)-1` 个字符，
  避免把 `<think>` / `</think>` 从中间切开。
- **前端不做整段重绘**：思考块直接改 `textContent`，正文按 ~55 ms 节流重渲染 markdown，
  这样折叠状态与选区不会在流式过程中被打断。
- **图表宽度跟 viewBox 同宽**（不是 `width:100%`）：矩阵 / 管线这类定尺寸图按自己的自然宽度
  绘制、左对齐，宿主放不下时收间距再收列宽，到下限才退回横向滚动 —— 而不是把 viewBox 压到
  宿主宽度再拉伸，那样图元会溢出卡片、在窄屏上被 `overflow:hidden` 直接裁掉。
- **窗口 resize 重建图表**：SVG 用 `viewBox`，但刻度密度依赖像素宽度，所以 `lab.js` 的
  `chartHost()` 把每次绘制登记进 `pendingCharts`，`resize` 后防抖 160 ms 全量重画。
