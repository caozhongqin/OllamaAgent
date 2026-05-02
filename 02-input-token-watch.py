"""
本地文件助手 Demo
依赖: pip install ollama
前提: ollama 已启动，且已拉取 qwen2.5:7b
"""

import json
from pathlib import Path
import ollama

client = ollama.Client(host="http://127.0.0.1:11434")
# ─────────────────────────────────────────────
# 0. 输入上下文定义（检查记忆使用情况防止溢出）
# ─────────────────────────────────────────────
NUM_CTX = 32768

def get_usage_value(resp, key: str, default=None):
    if hasattr(resp, key):
        return getattr(resp, key)

    if isinstance(resp, dict):
        return resp.get(key, default)

    try:
        return resp[key]
    except Exception:
        return default


def check_context_usage(resp, label=""):
    prompt_tokens = get_usage_value(resp, "prompt_eval_count")

    if prompt_tokens is None:
        print(f"[上下文监控] {label} 未获取到 prompt_eval_count")
        return "unknown"

    ratio = prompt_tokens / NUM_CTX

    print(
        f"[上下文监控] {label} "
        f"输入 tokens: {prompt_tokens} / {NUM_CTX} "
        f"({ratio:.2%})"
    )

    if ratio >= 0.92:
        print("[上下文监控] 极高风险：建议立即重启对话或强制压缩历史。")
        return "critical"

    if ratio >= 0.85:
        print("[上下文监控] 危险：建议立即生成摘要，并重建 messages。")
        return "danger"

    if ratio >= 0.75:
        print("[上下文监控] 警告：建议压缩历史，避免继续累积工具结果。")
        return "warn"

    if ratio >= 0.60:
        print("[上下文监控] 注意：上下文已经过半，可以开始减少无关历史。")
        return "notice"

    return "ok"

# ─────────────────────────────────────────────
# 1. 工具定义（告诉 llama 有哪些工具可以用）
# ─────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取本地文件的文本内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径，如 ./notes.txt"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入本地文件（可选覆盖或追加）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的文字内容"},
                    "mode":    {
                        "type": "string",
                        "description": "写入模式",
                        "enum": ["overwrite", "append"],
                        "default": "overwrite"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出某个目录下的所有文件和子文件夹",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径，默认为当前目录 '.'"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "删除指定的本地文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要删除的文件路径"}
                },
                "required": ["path"]
            }
        }
    }
]


# ─────────────────────────────────────────────
# 2. 工具实现（Python 真正执行的部分）
# ─────────────────────────────────────────────
def execute_tool(name: str, args: dict) -> str:
    """根据工具名和参数执行对应操作，返回结果字符串"""
    try:
        if name == "read_file":
            p = Path(args["path"])
            if not p.exists():
                return f"[错误] 文件不存在：{p.resolve()}"
            content = p.read_text(encoding="utf-8")
            return f"[读取成功] {p.resolve()}\n---\n{content}\n---"

        elif name == "write_file":
            p = Path(args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            mode = args.get("mode", "overwrite")
            if mode == "append":
                with open(p, "a", encoding="utf-8") as f:
                    f.write(args["content"])
                return f"[追加成功] {p.resolve()}"
            else:
                p.write_text(args["content"], encoding="utf-8")
                return f"[写入成功] {p.resolve()}"

        elif name == "list_directory":
            p = Path(args.get("path", "."))
            if not p.exists():
                return f"[错误] 目录不存在：{p.resolve()}"
            items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
            if not items:
                return "[目录为空]"
            lines = []
            for item in items:
                icon = "📄" if item.is_file() else "📁"
                lines.append(f"  {icon} {item.name}")
            return f"[目录] {p.resolve()}\n" + "\n".join(lines)

        elif name == "delete_file":
            p = Path(args["path"])
            if not p.exists():
                return f"[错误] 文件不存在：{p.resolve()}"
            p.unlink()
            return f"[删除成功] {p.resolve()}"

        else:
            return f"[错误] 未知工具：{name}"

    except Exception as e:
        return f"[执行异常] {name} → {e}"


# ─────────────────────────────────────────────
# 3. 主对话循环
# ─────────────────────────────────────────────
def chat_loop():
    print("\n" + "═" * 52)
    print("  🗂  本地文件助手  |  基于 qwen2.5:7b + ollama")
    print("  支持：读文件 / 写文件 / 列目录 / 删文件")
    print("  输入 exit 或 quit 退出")
    print("═" * 52 + "\n")

    messages = []  # 维护完整对话历史，让模型有记忆

    while True:
        # ── 获取用户输入 ──
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break

        messages.append({"role": "user", "content": user_input})

        # ── 第一次请求 llama（可能返回工具调用）──
        try:
            resp = client.chat(
                model="qwen2.5:7b",
                messages=messages,
                tools=TOOLS,
                options={
                    "num_ctx": NUM_CTX,
                    "num_predict": 2048
                }
            )
            context_status = check_context_usage(resp, "第一次请求")
        except Exception as e:
            print(f"[连接 ollama 失败] {e}\n请确认 ollama 已启动，且已拉取 qwen2.5:7b\n")
            messages.pop()  # 回滚这条消息
            continue
        # print(resp)  # 打印完整响应对象，方便调试
        # 纯聊天时（prompt_eval_count=340）
        # model='qwen2.5:7b' created_at='2026-05-02T03:01:50.815245Z' done=True done_reason='stop' 
        # total_duration=761590500 load_duration=98891500 
        # prompt_eval_count=340 prompt_eval_duration=268506500 
        # eval_count=8 eval_duration=382388900 message=Message(role='assistant', content='你好！有什么可以帮助你的吗？', thinking=None, images=None, 
        # tool_name=None, tool_calls=None) logprobs=None
        # 调用工具时（prompt_eval_count=365）
        # model='qwen2.5:7b' created_at='2026-05-02T03:12:07.3947772Z' done=True done_reason='stop' 
        # total_duration=3522150700 load_duration=2279081000 
        # prompt_eval_count=365 prompt_eval_duration=373191800 
        # eval_count=19 eval_duration=836110500 message=Message(role='assistant', content='', thinking=None, images=None, 
        # tool_name=None, tool_calls=[ToolCall(function=Function(name='list_directory', arguments={'path': '.'}))]) logprobs=None
        # 再次闲聊时（prompt_eval_count=561）
        # model='qwen2.5:7b' created_at='2026-05-02T03:14:18.1064613Z' done=True done_reason='stop' 
        # total_duration=1079151500 load_duration=102700300 
        # prompt_eval_count=561 prompt_eval_duration=433827500 
        # eval_count=11 eval_duration=525355100 message=Message(role='assistant', content='你好！有什么问题或者需要帮助的吗？', thinking=None, images=None, 
        # tool_name=None, tool_calls=None) logprobs=None
        # 总结：prompt_eval_count 随着交互的增多，输入的Token数量增加而增加；eval_count 则是模型实际推理的次数，调用工具时会增加，因为工具结果也会被模型重新评估。
        msg = resp.message

        # ── 如果 llama 决定调用工具 ──
        if msg.tool_calls:
            # 把助手这条"想调用工具"的消息加入历史
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls
            })

            # 逐个执行工具
            for call in msg.tool_calls:
                fn_name = call.function.name
                fn_args = call.function.arguments  # dict

                print(f"\n  🔧 调用工具：{fn_name}")
                print(f"     参数：{json.dumps(fn_args, ensure_ascii=False)}")

                result = execute_tool(fn_name, fn_args) # type: ignore
                print(f"  📋 结果：{result}\n")
                print(f"  📏 工具结果字符数：{len(result)}")

                # 把工具执行结果加入历史（role=tool）
                messages.append({
                    "role": "tool",
                    "content": result
                })

            # ── 第二次请求 llama（让它基于工具结果给出最终回复）──
            try:
                final_resp = client.chat(
                    model="qwen2.5:7b",
                    messages=messages,
                    tools=TOOLS,
                    options={
                        "num_ctx": NUM_CTX,
                        "num_predict": 2048
                    }
                )
                context_status = check_context_usage(final_resp, "第二次请求")
                final_content = final_resp.message.content or ""
                messages.append({"role": "assistant", "content": final_content})
                print(f"助手：{final_content}\n")
            except Exception as e:
                print(f"[第二次请求失败] {e}\n")

        # ── 普通文本回复（llama 判断不需要工具）──
        else:
            content = msg.content or ""
            messages.append({"role": "assistant", "content": content})
            print(f"助手：{content}\n")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    chat_loop()