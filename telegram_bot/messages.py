UNAUTHORIZED = "未授权：此 Telegram 用户不能控制本电脑。"
BUSY = "当前已有任务正在执行。可发送 /stop 立即停止。"
NO_TASK = "当前没有正在执行的 AI 任务。"


def command_usage(command: str, example: str) -> str:
    return f"用法：/{command} {example}"

