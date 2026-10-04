# AI PC Agent（Windows 11）

这是一个可直接运行在 Windows 11 交互式桌面会话中的本地 Agent。手机端通过 Telegram 发送命令或自然语言；本地模型负责理解和规划，本机工具层负责真实读取状态、启动应用、切换窗口和执行输入动作。

## 本地 Qwen 视觉上下文

八雀视觉请求包含截图，Ollama 默认 4096 context 容易出现：

```text
request (...) exceeds the available context size (4096 tokens)
```

AIAGENT 已提供 16K 模型配置。Windows 下在项目根目录运行：

```bat
scripts\setup_ollama_qwen16k.bat
```

脚本会：
- 基于本地 `qwen3.5:9b` 创建 `qwen3.5-9b-16k`；
- 设置 `PARAMETER num_ctx 16384`；
- 自动备份并更新现有 `.env` 中的 `OPENAI_MODEL` 与 `GAME_VISION_MODEL`；
- 不删除原来的 `qwen3.5:9b`。

完成后必须彻底停止并重新启动 `main.py`。如果只修改代码、不创建 16K 模型，4096 context 的视觉请求仍可能失败。

## AIAGENT 分支定位

该分支把项目从“某个游戏的自动化程序”调整为**通用电脑 AI 助手**：

```text
Telegram / 后续语音入口
        ↓
本地 Qwen3.5:9b / OpenAI
        ↓
Agent Planner + Tool Router
        ↓
Windows 工具层
  ├─ 应用：启动 / 检查运行 / 置前 / 关闭
  ├─ 窗口：枚举 / 获取前台 / 按标题置前
  ├─ 屏幕：截图
  ├─ 输入：鼠标 / 键盘
  ├─ 系统：CPU / 内存 / 进程
  └─ 插件：点字成金等专项能力
```

点字成金现在通过 `plugins/baque.py` 组装，`game_baque/` 保留原来的视觉、记牌、对手模型和 EV 引擎。主程序只把它当作一个插件，不再以游戏作为 Agent 核心。

当配置 `OPENAI_BASE_URL`（例如 Ollama）时，规划器自动走 OpenAI-compatible **Chat Completions + function calling**。对于部分本地模型偶尔把工具请求输出成 JSON 文本的情况，还提供“仅限已注册工具”的 JSON 兜底解析。因此模型收到“微信有没有运行”“打开抖音”时，应先实际调用本机工具，而不是回答“我无法查看你的设备”。

## 已实现功能

- Telegram 用户白名单：只有 `.env` 中指定的用户 ID 可以操作。
- `/start`、`/status`、`/screen`、`/mouse`、`/click`、`/doubleclick`、`/move`、`/scroll`、`/type`、`/hotkey`、`/process`、`/stop`。
- 点字成金：`/game_state`、`/game_step`、`/game_start`、`/game_stop`。
- 牌局采集：`/game_record_start`、`/game_record_status`、`/game_record_stop`。
- OpenAI 官方接口使用 Responses API；配置本地 `OPENAI_BASE_URL` 时使用 Chat Completions/function calling。最多 15 步，不执行模型生成的 Python、PowerShell、CMD、`eval`、`exec` 或任意 Shell。
- 本地白名单应用启动、运行状态检测、窗口置前、确认后关闭、网页打开、截图、系统状态、鼠标键盘和受管进程。
- 自然语言阶段暴露 22 个通用电脑工具 + 10 个八雀插件工具；窗口枚举/置前使用 Windows UI Automation，但更高风险的通用 UIA 点击、文件和 Git 工具仍不向模型开放。
- 当前不启用视觉 AI 坐标推断；坐标类操作必须由用户明确给出或使用确定性计算。
- 高 DPI 感知、PyAutoGUI FailSafe、`Ctrl+Alt+F12` 紧急停止、单 AI 任务互斥。
- 高风险文件、Shell、生产系统和资金操作没有注册工具，模型无法调用。
- 日志滚动与密钥脱敏。运行日志统一保存到项目根目录 `LOGS/`；单个工具异常只返回失败结果，不会终止主程序。

## 通用自然语言 Agent

普通 Telegram 文本会交给 Planner。当前注册 22 个通用电脑工具，并在八雀插件可用时额外注册 10 个八雀工具：

```text
launch_app             is_app_running
focus_app              close_app
list_windows           focus_window
take_screenshot        get_system_status
get_active_window      get_mouse_position
move_mouse             click
double_click           scroll
type_text              press_key
hotkey                 open_url
list_processes         start_process
stop_managed_process   wait
```

八雀插件工具：

```text
baque_get_state          baque_step
baque_watch_start        baque_watch_status
baque_watch_stop         baque_auto_start
baque_auto_stop          baque_acceptance_report
baque_acceptance_mark    baque_acceptance_reset
```

因此可以直接发送自然语言：

```text
看看八雀现在什么状态
我自己玩，你只帮我记三家的牌
看看三家现在可能在做什么牌型
八雀只走一步
开始八雀自动游戏
停止八雀自动游戏
刚才自己的牌面识别对了
刚才对手那次记牌错了，Q黑桃其实是Q红桃
看看八雀目前准确率够不够无人值守
```

应用白名单位于 `config/apps.yaml`。AIAGENT 分支包含 Chrome、Edge、微信、抖音、记事本、资源管理器、CMD、PowerShell、IDEA 和 `a_share_recommend`，并支持“微信/weixin”“抖音/douyin”等别名。`cmd`、`powershell` 仅允许打开空窗口，AI 不能传入命令；`start_process` 的 `program` 也必须是 YAML 中的应用名称，不能传入原始可执行路径。

可直接测试以下消息：

```text
给我看看电脑现在的屏幕
微信有没有运行
打开微信
切到微信
打开抖音
打开 Chrome
打开记事本
把鼠标移动到屏幕中央
打开百度
查看当前运行的 Python 程序
启动 a_share_recommend
告诉我电脑现在怎么样
```

任务状态为 `IDLE`、`RUNNING`、`WAITING_CONFIRMATION`、`STOPPING`。同一时间只运行一个任务；`/stop` 会立即取消模型请求或可中断等待，已经进入 Windows 原生调用的操作会安全收尾后停止，不会继续执行下一工具。

删除文件、删除目录、关机、重启、格式化磁盘、安装软件、发送邮件/微信、付款、转账、真实股票下单、生产数据库修改、敏感文件上传等功能在第一版**没有注册工具，因此无法执行**。模型也被明确禁止利用鼠标或浏览器工具绕过这一限制。

## 1. 安装 Python

安装 64 位 Python 3.11 或更高版本，并在安装界面勾选“Add Python to PATH”。在 PowerShell 中确认：

```powershell
py -3 --version
```

## 2. 创建 Telegram Bot

1. 在 Telegram 搜索 `@BotFather`。
2. 发送 `/newbot`，按提示设置名称。
3. 保存 BotFather 返回的 Token。不要把 Token 发给其他人或提交到 Git。

## 3. 获取 Telegram User ID

在 Telegram 搜索 `@userinfobot`（或其他可信的 User ID 查询 Bot），向它发送任意消息，记录返回的数字 ID。项目使用的是用户 ID，不是用户名。

## 4. 配置 `.env`

在项目目录执行：

```powershell
Copy-Item .env.example .env
notepad .env
```

填写：

```dotenv
TELEGRAM_BOT_TOKEN=BotFather给出的Token
ALLOWED_TELEGRAM_USER_ID=你的数字UserID
OPENAI_API_KEY=你的OpenAI_API_Key
OPENAI_MODEL=qwen3.5:9b
OPENAI_BASE_URL=
AGENT_ALLOWED_ROOTS=D:\Projects;D:\Work
```

应用路径和启动参数统一在 `config/apps.yaml` 中维护。模型名称可按账号实际可用模型修改；固定 Telegram 指令不需要模型。自然语言 Agent 使用 OpenAI 官方接口时配置 `OPENAI_API_KEY`，使用本地兼容接口时配置 `OPENAI_BASE_URL`。

如果使用本机 Ollama，可改为：

```dotenv
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen3.5:9b
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
```

Ollama 不校验 `OPENAI_API_KEY`，这里的 `ollama` 只是 OpenAI SDK 要求的占位值。项目通过 Telegram long polling 收到消息后，会把普通文本交给该本地接口；`/status`、`/screen` 等固定指令不经过模型。配置了 `OPENAI_BASE_URL` 后会自动使用 Chat Completions/function calling，适配本地 `qwen3.5:9b`。

## 5. 安装依赖

双击 `start.bat` 会自动创建 `.venv` 并安装依赖。也可手动执行：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 6. 安装 Playwright Chromium

首次使用 Playwright 前执行：

```powershell
.\.venv\Scripts\Activate.ps1
playwright install chromium
```

## 7. 启动

直接双击 `start.bat`，或：

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

看到 `Telegram: Connected` 和 `Agent: READY` 后，在手机上先依次测试 `/start`、`/status`、`/screen`，再测试 `/click 800 500`。最后发送“打开 Chrome”测试自然语言工具调用。

## 8. Windows 权限问题

- Agent 只能操作与它权限级别相同或更低的窗口。若目标程序以管理员身份运行，Agent 也需“以管理员身份运行”。
- 不建议日常以管理员身份启动；权限越高，误操作影响越大。
- Windows 锁屏、UAC 安全桌面、断开的 RDP 会话通常不能截图或模拟输入。
- UI Automation 对不同版本 IDEA/Chrome 的控件树可能不同，可先用自然语言要求 Agent 查找控件。

## 9. 高 DPI 与多显示器

程序在导入 GUI 库前启用 Per-Monitor DPI Awareness，截图坐标使用 PNG 原始像素。第一版已校验坐标范围，但主要按 PyAutoGUI 提供的桌面尺寸工作；混合缩放、多显示器位于主屏左侧（负坐标）等复杂布局需要人工实测。关键操作前建议先 `/screen`，确认截图与坐标一致。

## 10. PyAutoGUI FailSafe

`pyautogui.FAILSAFE = True`。将鼠标快速移动到主屏左上角会触发 FailSafe，使当前鼠标工具失败并将错误返回 Agent。也可以按 `Ctrl+Alt+F12`，或在 Telegram 发送 `/stop`。

`/stop` 是协作式停止：它会立即设置停止信号并取消待确认操作；如果 Windows 原生调用已经开始，会在该次调用返回后安全释放任务锁，避免旧任务与新任务同时控制鼠标。

## 日志目录

程序运行日志统一保存到：

```text
LOGS/
├── agent.log
├── agent.log.1
├── agent.log.2
└── baque/
    ├── acceptance_labels.jsonl
    └── acceptance_report.json
```

其中 `agent.log` 使用滚动日志；控制台普通 INFO/WARNING 日志输出到 stdout，IDE 中保持普通白色；只有 ERROR/CRITICAL 输出到 stderr，因此显示为红色。Telegram 收到的每一条文本命令和自然语言指令都会先写入 `LOGS/agent.log`，格式包含时间、用户 ID、聊天 ID 和原始指令文本。换行会转义为 `\\n`，单条超长指令会截断，已配置的 Token/API Key 仍会经过脱敏过滤。工具执行和工具结果也继续写入同一运行日志。八雀的验收真值与验收报告统一放到 `LOGS/baque/`。牌局截图、关键帧和采集素材仍保存在 `game_recordings/`，因为它们属于调试/训练数据而不是运行日志。

## 11. 安全注意事项

- `.env`、日志、截图均被 `.gitignore` 排除；截图仍可能含敏感信息，请定期清理。
- 不要将项目目录、Token、API Key 暴露给不可信用户。
- `start_process` 只允许白名单程序。Python 只允许执行允许目录中的 `.py` 文件，禁止 `python -c` 和 `python -m`。
- `run_safe_command` 不使用 `shell=True`，拒绝 Shell 元字符，并限制 Git 子命令。
- 文件 API 只允许访问 `AGENT_ALLOWED_ROOTS`。文本读取会截断，命令输出也有长度限制。
- 通用鼠标点击无法理解按钮背后的业务含义。涉及资金、生产系统或敏感信息时不要依赖通用坐标自动化。

## 12. 常见错误

- **`.env 缺少配置`**：确认文件名确实是 `.env`，不是 `.env.txt`。
- **Unauthorized / 未授权**：检查 User ID 是否为纯数字且属于当前 Telegram 账号。
- **Conflict: terminated by other getUpdates request**：同一个 Bot Token 有另一个实例正在运行，关闭旧实例。
- **Chromium executable doesn't exist**：执行 `playwright install chromium`。
- **未找到 Chrome/IDEA**：当前版本只查找常见安装位置。可在 `tools/browser.py` 增加本机绝对路径。
- **截图黑屏或失败**：确认处于已登录、未锁屏的本地桌面会话；某些远程桌面断开后无法捕获。
- **中文输入失败**：`/type` 会通过剪贴板粘贴 Unicode；目标控件阻止粘贴时，改用 UIA 输入。
- **OpenAI 400/model not found**：将 `.env` 的 `OPENAI_MODEL` 改成账号可用且支持工具调用的模型。

## 13. 设置 Windows 开机启动

推荐使用“任务计划程序”，并选择“仅当用户登录时运行”（GUI 自动化必须有交互式桌面）：

1. 打开“任务计划程序”→“创建任务”。
2. 触发器选择“登录时”。
3. 操作选择“启动程序”。
4. 程序填写项目中的 `start.bat` 完整路径。
5. “起始于”填写项目目录。
6. 默认不要勾选“使用最高权限运行”。

## 测试与基础自检

```powershell
.\.venv\Scripts\Activate.ps1
python -m compileall .
pytest
```

截图测试在没有可交互桌面的环境中会自动跳过。Telegram 和 OpenAI 的真实连通性需要填写 `.env` 后通过 `/start`、`/screen` 和一条自然语言命令验证。

## 项目结构

```text
ai_pc_agent/
├── agent/             # 规划、执行、工具注册、安全、会话
├── telegram_bot/      # 指令、消息、确认回调
├── tools/             # Windows/浏览器/文件/Git 工具
├── tests/             # 安全、步骤上限、停止、截图等测试
├── main.py
├── config.py
├── emergency.py
├── requirements.txt
├── .env.example
└── start.bat
```

## 第一版明确限制

- 不包含 OCR、任意 JavaScript、任意 Shell、文件删除、文件上传和生产系统专用工具。
- Playwright 使用独立受控 Chromium，不复用个人 Chrome 登录态。
- UIA 调用无法安全强杀；已设置工具超时的地方会优先超时，`/stop` 会等待正在运行的原生调用收尾。
- Windows 应用版本、显示器布局和网页 DOM 差异仍需在目标电脑上做一次真实验收。


## 点字成金自动游戏（实验功能）

该模块针对腾讯欢乐斗地主中的“百变八雀·点字成金”。实现方式仅为截图识别 + 普通鼠标点击，不读取游戏进程内存、不注入 DLL、不修改客户端、不绕过反作弊。

工作流程：

    欢乐斗地主前台窗口
    → 截取游戏窗口
    → 本地/远程视觉模型识别金牌/手牌/亮牌/按钮/当前阶段与对手事件
    → 本地 rules.py 判断 19 种胡牌牌型
    → 本地 strategy.py 决定换牌、摸牌、出牌或胡牌
    → PyAutoGUI 点击
    → 等待新画面后继续

规则引擎当前按项目约定实现：点数为 6、7、8、9、10、J、Q、K、A；6~10 为 10 分，J/Q/K 与癞子 20 分，A 30 分；金牌数 0~1/2/3/4/5/6+ 对应倍率 1/2/4/8/16/32；支持 8 张同点/同花顺、6+2、5+3、4+4、3+3+2，以及癞子补牌。当前暂按硬气胡额外 ×2 处理。

Telegram 命令：

- /game_state：只截图识别并返回状态，不点击。第一次务必先用它核对识别结果。
- /game_step：只执行一个动作，适合调试。
- /game_start：连续执行。启动时会先自动切到欢乐斗地主窗口；若当前停在百变八雀的“开始游戏”页面，会识别并自动点击“开始游戏”，等待正式发牌后再启动三家高频观察器和 EV。
- /game_stop：停止连续自动游戏和本次对手观察。
- /game_watch_start：只启动三家实时观察/记牌，不自动点击，适合先手动玩一局验证。
- /game_watch_status：查看采样帧、动作片段、模型识别结果和当前三家记牌/危险牌推断。
- /game_watch_stop：停止三家实时观察器。
- /game_acceptance：查看当前真实牌局验收报告。
- /game_accept own|opponent|action ok|bad [备注]：给刚才结果打真值标签。
- /game_accept_reset：开始新一轮验收统计，历史标签仍保留。

在 .env 可配置：

    GAME_VISION_MODEL=qwen3.5:9b
    GAME_VISION_BASE_URL=http://127.0.0.1:11434/v1
    GAME_WINDOW_TITLE=欢乐斗地主
    GAME_ACTION_INTERVAL_SECONDS=1.5
    GAME_MIN_CARD_CONFIDENCE=0.72
    GAME_MIN_EVENT_CONFIDENCE=0.80
    GAME_AUTO_NEXT_ROUND=false

GAME_AUTO_NEXT_ROUND 默认关闭。一局结束后停止；检测到“豆子不足/充值/购买”弹窗也会直接停止，不会自动购买豆子、钻石或执行充值。

自动模式现在不再要求你手动把欢乐斗地主切到前台：即使你发命令时前台是抖音、IDEA 等窗口，`/game_start` / “开始八雀自动游戏”也会先自动聚焦欢乐斗地主。只有视觉模型明确识别到橙色“开始游戏”按钮时才会点击；不会用固定坐标盲点。

第一次真实测试顺序：

1. 启动游戏并进入点字成金牌桌，保持游戏窗口在前台。
2. Telegram 发送 /game_state。
3. 核对 3 个金牌、7/8 张手牌、花色和亮牌是否全部正确。
4. 发送 /game_watch_start，先手动玩一局；期间用 /game_watch_status 检查左/上/右三家的摸牌、拿亮牌和弃牌是否正确写入 TableTracker。
5. 对手观察稳定后停止 /game_watch_stop，再用 /game_step 一次只执行一步，核对 EV 决策与点击。
6. 单步稳定后最后才使用 /game_start；自动模式会同时启动高频对手观察器。

当前已支持通过 Ollama 使用本地 `qwen3.5:9b` 进行视觉识别，不需要远程视觉 API。自动模式包含两条并行链路：主链识别自己的手牌/按钮并计算动作；后台观察器以约 GAME_RECORD_FPS 的频率做本地图像差分，把连续动画合并成 BEFORE/PEAK/AFTER 三关键帧，只在检测到真实动作片段时调用 Qwen3.5。高置信度的 left/top/right 完整回合会原子写入 TableTracker，EV 使用当时牌局状态的独立快照计算，因此不会阻塞后台采样，也不会在计算过程中被新事件改写牌池。

### 真实牌局验收

仅仅“程序没有报错”不能证明视觉识别正确，因此 AIAGENT 增加了两层验收：

1. **自动质量指标**：统计画面识别次数、unknown 率、可操作画面拒绝率、对手动作片段/低置信率以及 TableTracker 牌池一致性。
2. **人工真值准确率**：你只需要在关键样本后告诉助手“刚才对/错”，程序会把真值追加写入 `LOGS/baque/acceptance_labels.jsonl`，重启后仍可追溯。

默认 PASS 门槛：

- 自己牌面：至少 20 个真值样本，准确率 ≥ 98%。
- 对手动作：至少 30 个真值样本，准确率 ≥ 95%。
- 决策/点击：至少 20 个真值样本，准确率 ≥ 99%。
- 至少 20 次画面识别，unknown 率 ≤ 10%。
- 至少 30 个已分类对手动作，低置信率 ≤ 20%。
- TableTracker 牌池计数必须无冲突。

在样本不足时报告为 `INSUFFICIENT_DATA`；任一硬指标失败为 `FAIL`；全部满足才会给 `PASS`。报告同时写入 `LOGS/baque/acceptance_report.json`。

建议真实测试时直接用自然语言反馈：

```text
刚才自己的牌识别对了
刚才对手动作识别错了，实际是右家拿了亮牌
刚才点击是对的
看看八雀验收结果
```



## 点字成金牌局采集模式

为了训练三家对手的实时动作识别器，先使用“只观察、不自动点击”的牌局采集模式。

### 使用方法

1. 打开欢乐斗地主并进入点字成金牌桌。
2. 保持游戏窗口在前台。
3. Telegram 发送 `/game_record_start`。
4. 正常手动玩完整一局。采集器会在本机高频观察游戏窗口，不调用 OpenAI，不控制鼠标。
5. 随时发送 `/game_record_status` 查看状态。
6. 一局结束后发送 `/game_record_stop`。

数据保存到 `game_recordings/YYYYMMDD_HHMMSS/`，包含：

- `meta.json`：本次采集参数和窗口信息。
- `summary.json`：采样帧、归档帧、事件候选数量等汇总。
- `frames.jsonl`：每次采样的整体变化与各区域变化分数。
- `events.jsonl`：事件候选、变化区域、前后关键帧路径。
- `frames/`：低频连续归档帧，默认约 2 FPS。
- `events/`：ROI 图像差分检测到明显变化时保存的高质量前后关键帧。

当前 `events.jsonl` 的 `label` 暂为空。下一阶段会把事件自动分类为左家/上家/右家的暗摸、拿亮牌、出牌、机会牌等动作。

采集期间如果切到其他窗口，程序会暂停采样，不会把其他软件截图写入牌局数据；切回欢乐斗地主后自动继续。

### 采集配置

`.env` 可调整：

    GAME_RECORD_FPS=6
    GAME_RECORD_ARCHIVE_FPS=2
    GAME_RECORD_DIFF_THRESHOLD=0.028
    GAME_RECORD_JPEG_QUALITY=78

第一次建议保持默认值。`game_recordings/` 已加入 `.gitignore`，真实牌局截图不会提交到 GitHub。

采集模式不依赖 OpenAI API，因此即使视觉模型尚未配置，也可以先采集真实牌局。

## 三家对手实时观察（基于真实牌局校准）

2026-10-04 首轮真实牌局样本：约 250.7 秒、1504 个 6 FPS 采样点、145 个 raw 变化事件。复盘发现大量 raw event 实际属于同一段动画/同一轮动作；按约 1.2 秒间隔聚类只有约 39 个动作簇。因此实时逻辑已经改为：

    6 FPS 本地截屏
    → 窄 ROI 差分
    → 合并为一个 MotionEpisode
    → 保存 BEFORE / PEAK / AFTER 三张关键帧
    → 过滤“硬气/比翼为邻/金倍率”等全屏特效
    → 仅对动作片段调用视觉模型
    → 高置信对手事件写入 TableTracker
    → 更新三家可能牌型、危险牌和 EV

首轮样本同时用于收紧 left/top/right 的手牌扇区 ROI，避免中央大特效同时污染三家区域。

### 手动观察测试

进入一局点字成金并保持游戏窗口在前台，先不要开启自动点击：

    /game_watch_start

程序会先读取自己的手牌、金牌和牌堆建立基线，然后后台观察三名对手。可随时查看：

    /game_watch_status

状态会显示动作片段数量、已识别数量、已写入记牌器数量、低置信事件、识别队列，以及三家当前已知存牌/未知牌数/可能牌型。

结束后：

    /game_watch_stop

调试文件保存在 `game_recordings/live_watch_时间/`：

- `episodes.jsonl`：本地动作片段及 ROI 峰值。
- `episodes/*_before.jpg`、`*_peak.jpg`、`*_after.jpg`：三关键帧。
- `recognized_events.jsonl`：视觉模型的结构化识别、是否写入记牌器、错误信息。
- `observer_summary.json`：整局观察汇总。

`/game_start` 进入连续自动游戏时，也会自动启动同一套对手观察器；建议先用 `/game_watch_start` 完整手动验证一两局再开启自动执行。
