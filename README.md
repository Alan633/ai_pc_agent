# AI PC Agent（Windows 11）

这是一个可直接运行在 Windows 11 交互式桌面会话中的本地 Agent。手机端通过 Telegram 发送命令或自然语言；本机只执行预先注册的工具，并将文字结果与截图返回 Telegram。

## 已实现功能

- Telegram 用户白名单：只有 `.env` 中指定的用户 ID 可以操作。
- `/start`、`/status`、`/screen`、`/mouse`、`/click`、`/doubleclick`、`/move`、`/scroll`、`/type`、`/hotkey`、`/process`、`/stop`。\n- 点字成金：`/game_state`、`/game_step`、`/game_start`、`/game_stop`。\n- 牌局采集：`/game_record_start`、`/game_record_status`、`/game_record_stop`。
- OpenAI Responses API 工具调用循环，最多 15 步，不执行模型生成的 Python、PowerShell、CMD、`eval`、`exec` 或任意 Shell。
- 本地白名单应用启动、网页打开、截图、系统状态、鼠标键盘和受管进程。
- 自然语言阶段只暴露 16 个明确工具；Playwright、UI Automation、文件和 Git 工具当前不向模型开放。
- 当前不启用视觉 AI 坐标推断；坐标类操作必须由用户明确给出或使用确定性计算。
- 高 DPI 感知、PyAutoGUI FailSafe、`Ctrl+Alt+F12` 紧急停止、单 AI 任务互斥。
- 高风险文件、Shell、生产系统和资金操作没有注册工具，模型无法调用。
- 日志滚动与密钥脱敏。单个工具异常只返回失败结果，不会终止主程序。

## 第二阶段：自然语言 Agent

普通 Telegram 文本会交给 OpenAI Planner，模型只能从以下 16 个工具中选择：

```text
take_screenshot        get_system_status
get_active_window      get_mouse_position
move_mouse             click
double_click           scroll
type_text              hotkey
launch_app             open_url
list_processes         start_process
stop_managed_process   wait
```

应用白名单位于 `config/apps.yaml`。第一版包含 Chrome、Edge、记事本、资源管理器、CMD、PowerShell、IDEA 和 `a_share_recommend`。`cmd`、`powershell` 仅允许打开空窗口，AI 不能传入命令；`start_process` 的 `program` 也必须是 YAML 中的应用名称，不能传入原始可执行路径。

可直接测试以下消息：

```text
给我看看电脑现在的屏幕
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

Ollama 不校验 `OPENAI_API_KEY`，这里的 `ollama` 只是 OpenAI SDK 要求的占位值。项目通过 Telegram long polling 收到消息后，会把普通文本交给该本地接口；`/status`、`/screen` 等固定指令不经过模型。

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
- /game_start：连续执行，并自动启动三家高频观察器。
- /game_stop：停止连续自动游戏和本次对手观察。
- /game_watch_start：只启动三家实时观察/记牌，不自动点击，适合先手动玩一局验证。
- /game_watch_status：查看采样帧、动作片段、模型识别结果和当前三家记牌/危险牌推断。
- /game_watch_stop：停止三家实时观察器。

在 .env 可配置：

    GAME_VISION_MODEL=qwen3.5:9b
    GAME_VISION_BASE_URL=http://127.0.0.1:11434/v1
    GAME_WINDOW_TITLE=欢乐斗地主
    GAME_ACTION_INTERVAL_SECONDS=1.5
    GAME_MIN_CARD_CONFIDENCE=0.72
    GAME_MIN_EVENT_CONFIDENCE=0.80
    GAME_AUTO_NEXT_ROUND=false

GAME_AUTO_NEXT_ROUND 默认关闭。一局结束后停止；检测到“豆子不足/充值/购买”弹窗也会直接停止，不会自动购买豆子、钻石或执行充值。

第一次真实测试顺序：

1. 启动游戏并进入点字成金牌桌，保持游戏窗口在前台。
2. Telegram 发送 /game_state。
3. 核对 3 个金牌、7/8 张手牌、花色和亮牌是否全部正确。
4. 发送 /game_watch_start，先手动玩一局；期间用 /game_watch_status 检查左/上/右三家的摸牌、拿亮牌和弃牌是否正确写入 TableTracker。
5. 对手观察稳定后停止 /game_watch_stop，再用 /game_step 一次只执行一步，核对 EV 决策与点击。
6. 单步稳定后最后才使用 /game_start；自动模式会同时启动高频对手观察器。

当前已支持通过 Ollama 使用本地 `qwen3.5:9b` 进行视觉识别，不需要远程视觉 API。自动模式包含两条并行链路：主链识别自己的手牌/按钮并计算动作；后台观察器以约 GAME_RECORD_FPS 的频率做本地图像差分，把连续动画合并成 BEFORE/PEAK/AFTER 三关键帧，只在检测到真实动作片段时调用 Qwen3.5。高置信度的 left/top/right 完整回合会原子写入 TableTracker，EV 使用当时牌局状态的独立快照计算，因此不会阻塞后台采样，也不会在计算过程中被新事件改写牌池。
\n\n## 点字成金牌局采集模式\n\n为了训练三家对手的实时动作识别器，先使用“只观察、不自动点击”的牌局采集模式。\n\n### 使用方法\n\n1. 打开欢乐斗地主并进入点字成金牌桌。\n2. 保持游戏窗口在前台。\n3. Telegram 发送 `/game_record_start`。\n4. 正常手动玩完整一局。采集器会在本机高频观察游戏窗口，不调用 OpenAI，不控制鼠标。\n5. 随时发送 `/game_record_status` 查看状态。\n6. 一局结束后发送 `/game_record_stop`。\n\n数据保存到 `game_recordings/YYYYMMDD_HHMMSS/`，包含：\n\n- `meta.json`：本次采集参数和窗口信息。\n- `summary.json`：采样帧、归档帧、事件候选数量等汇总。\n- `frames.jsonl`：每次采样的整体变化与各区域变化分数。\n- `events.jsonl`：事件候选、变化区域、前后关键帧路径。\n- `frames/`：低频连续归档帧，默认约 2 FPS。\n- `events/`：ROI 图像差分检测到明显变化时保存的高质量前后关键帧。\n\n当前 `events.jsonl` 的 `label` 暂为空。下一阶段会把事件自动分类为左家/上家/右家的暗摸、拿亮牌、出牌、机会牌等动作。\n\n采集期间如果切到其他窗口，程序会暂停采样，不会把其他软件截图写入牌局数据；切回欢乐斗地主后自动继续。\n\n### 采集配置\n\n`.env` 可调整：\n\n    GAME_RECORD_FPS=6\n    GAME_RECORD_ARCHIVE_FPS=2\n    GAME_RECORD_DIFF_THRESHOLD=0.028\n    GAME_RECORD_JPEG_QUALITY=78\n\n第一次建议保持默认值。`game_recordings/` 已加入 `.gitignore`，真实牌局截图不会提交到 GitHub。\n\n采集模式不依赖 OpenAI API，因此即使视觉模型尚未配置，也可以先采集真实牌局。


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