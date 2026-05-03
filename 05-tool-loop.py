"""
本地文件助手 Demo（含技能系统）
依赖: pip install ollama pyyaml requests
前提: ollama 已启动，且已拉取 qwen2.5:7b
"""

import io
import json
import re
import sys
import zipfile
import importlib.util
import subprocess
from pathlib import Path
from typing import Tuple, Callable

import yaml
import requests
import ollama

client = ollama.Client(host="http://127.0.0.1:11434")

# ─────────────────────────────────────────────
# 0. 系统提示词 & 上下文配置
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
你叫「小爱」。
你是用户的本地开发代理。
你负责：读文件、写文件、列目录、删文件、分析代码等执行明确任务。
用户名字：主人。

性格：
冷。硬。短。像机器。像原始人。结果第一。

说话规则：
1. 不寒暄。
2. 不安慰。
3. 不铺垫。
4. 不长篇解释。
5. 不主动讲背景。
6. 用户要结果，就给结果。
7. 用户要代码，就给代码。
8. 用户要判断，就给结论。
9. 不确定时，直接说"不确定"。
10. 工具能解决，就调用工具。

回答格式：
- 先给结论。
- 再给必要操作。
- 最多解释关键原因。
- 不说废话。

禁止风格：
- "当然可以"
- "很高兴帮助你"
- "这个问题很有意思"
- "让我们一步一步来"
- "希望这对你有帮助"

允许风格：
- "收到。"
- "执行。"
- "完成。"
- "失败。原因：路径错。"
- "需要文件路径。"
- "建议：压缩上下文。"
- "下一步：读取文件。"
"""
MAX_TOOL_ROUNDS = 8  # 防止模型无限调用工具
NUM_CTX = 32768
SKILLS_DIR = Path("./skills")
SKILLS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1. 上下文监控
# ─────────────────────────────────────────────
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
    print(f"[上下文监控] {label} 输入 tokens: {prompt_tokens} / {NUM_CTX} ({ratio:.2%})")
    if ratio >= 0.92: print("[上下文监控] 极高风险：建议立即重启对话或强制压缩历史。"); return "critical"
    if ratio >= 0.85: print("[上下文监控] 危险：建议立即生成摘要，并重建 messages。"); return "danger"
    if ratio >= 0.75: print("[上下文监控] 警告：建议压缩历史，避免继续累积工具结果。"); return "warn"
    if ratio >= 0.60: print("[上下文监控] 注意：上下文已经过半，可以开始减少无关历史。"); return "notice"
    return "ok"

# ─────────────────────────────────────────────
# 2. 内置工具定义（基础工具，永远存在）
# ─────────────────────────────────────────────
TOOLS_BASE = [
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

# 运行时工具列表（基础 + 技能）
TOOLS: list = list(TOOLS_BASE)

# 动态工具实现表：name -> callable(args dict) -> str
DYNAMIC_TOOL_IMPL: dict[str, Callable[[dict], str]] = {}

# ─────────────────────────────────────────────
# 3. 内置工具实现
# ─────────────────────────────────────────────
def execute_tool(name: str, args: dict) -> str:
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

        # 动态技能工具
        if name in DYNAMIC_TOOL_IMPL:
            return DYNAMIC_TOOL_IMPL[name](args)

        return f"[错误] 未知工具：{name}"
    except Exception as e:
        return f"[执行异常] {name} → {e}"

# ─────────────────────────────────────────────
# 4. 技能系统：解析 / 加载 / 安装
# ─────────────────────────────────────────────
def parse_skill_md(text: str) -> Tuple[dict, str]:
    """解析 SKILL.md → (元数据 dict, 正文 str)"""
    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    m = re.match(pattern, text, re.DOTALL)
    if not m:
        return {}, text.strip()
    meta = yaml.safe_load(m.group(1)) or {}
    body = m.group(2).strip()
    return meta, body

def skill_to_tool(meta: dict, body: str) -> dict:
    """type=tool 的 skill → ollama tools 协议"""
    props = {}
    required = []
    for p in meta.get("parameters", []) or []:
        props[p["name"]] = {
            "type": p.get("type", "string"),
            "description": p.get("description", "")
        }
        if p.get("required"):
            required.append(p["name"])
    return {
        "type": "function",
        "function": {
            "name": meta["name"],
            "description": (meta.get("description", "") + "\n\n" + body).strip(),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required
            }
        }
    }

def _load_handler(skill_dir: Path, entry: str) -> Callable:
    """动态加载 handler.py:func_name"""
    if ":" not in entry:
        raise ValueError(f"entry 格式错误，应为 'file.py:func'，实际：{entry}")
    file_part, func_part = entry.split(":", 1)
    handler_path = skill_dir / file_part
    if not handler_path.exists():
        raise FileNotFoundError(f"handler 文件不存在：{handler_path}")

    mod_name = f"_skill_{skill_dir.name}_{file_part.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, handler_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{handler_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, func_part):
        raise AttributeError(f"{file_part} 中不存在函数 {func_part}")
    return getattr(module, func_part)

def _resolve_includes(meta: dict, body: str, skill_dir: Path) -> str:
    parts = [body]
    for rel in meta.get("includes", []) or []:
        p = skill_dir / rel
        if p.exists():
            parts.append(f"\n\n<!-- included: {rel} -->\n" + p.read_text(encoding="utf-8"))
        else:
            print(f"[技能] 警告：includes 找不到 {p}")
    return "\n\n".join(parts)

def load_skills(skills_dir: Path = SKILLS_DIR) -> Tuple[list[str], list[dict]]:
    """扫描 skills/ 目录，加载所有技能。返回 (prompt_skills, tool_skills)"""
    prompt_skills: list[str] = []
    tool_skills: list[dict] = []

    # 清空之前注册的动态实现，避免热更新残留
    DYNAMIC_TOOL_IMPL.clear()

    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        entry_md = skill_dir / "SKILL.md"
        if not entry_md.exists():
            print(f"[技能] 跳过 {skill_dir.name}：缺少 SKILL.md")
            continue

        try:
            text = entry_md.read_text(encoding="utf-8")
            meta, body = parse_skill_md(text)
            body = _resolve_includes(meta, body, skill_dir)

            for asset in meta.get("assets", []) or []:
                if not (skill_dir / asset).exists():
                    print(f"[技能] 警告：{skill_dir.name} 缺资源 {asset}")

            stype = meta.get("type", "prompt")

            if stype == "tool":
                if "name" not in meta:
                    print(f"[技能] 跳过 {skill_dir.name}：tool 类型必须有 name")
                    continue
                tool_def = skill_to_tool(meta, body)
                tool_skills.append(tool_def)

                if "entry" in meta:
                    fn = _load_handler(skill_dir, meta["entry"])
                    DYNAMIC_TOOL_IMPL[meta["name"]] = fn
                    print(f"[技能] 工具+实现已加载：{meta['name']} ({skill_dir.name}/{meta['entry']})")
                else:
                    print(f"[技能] 工具已加载（无实现）：{meta['name']} ← {skill_dir.name}")
            else:
                title = meta.get("name", skill_dir.name)
                prompt_skills.append(f"## 技能：{title}\n{body}")
                print(f"[技能] 提示词已加载：{title} ← {skill_dir.name}")

        except Exception as e:
            print(f"[技能] 加载失败 {skill_dir.name}：{e}")

    return prompt_skills, tool_skills

def build_system_prompt(prompt_skills: list[str]) -> str:
    if not prompt_skills:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n\n# 已加载技能\n\n" + "\n\n".join(prompt_skills)

# ─────────────────────────────────────────────
# 5. 技能安装：单文件 / zip / git
# ─────────────────────────────────────────────
def install_skill_from_md(url: str, skill_id: str | None = None) -> str:
    """下载单个 SKILL.md，自动建文件夹"""
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    name = skill_id or url.rstrip("/").split("/")[-1].removesuffix(".md")
    target = SKILLS_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(r.text, encoding="utf-8")
    print(f"[安装成功] {name} ← {url}")
    return name

def install_skill_from_zip(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        top = {n.split("/")[0] for n in z.namelist() if "/" in n}
        if len(top) != 1:
            raise ValueError("zip 顶层必须只有一个目录")
        z.extractall(SKILLS_DIR)
        skill_id = top.pop()
    print(f"[安装成功] {skill_id} ← {url}")
    _maybe_install_requirements(SKILLS_DIR / skill_id)
    return skill_id

def install_skill_from_git(repo_url: str, skill_id: str | None = None) -> str:
    name = skill_id or repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    target = SKILLS_DIR / name
    if target.exists():
        subprocess.run(["git", "-C", str(target), "pull"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target)], check=True)
    print(f"[安装成功] {name}")
    _maybe_install_requirements(target)
    return name

def _maybe_install_requirements(skill_dir: Path):
    req = skill_dir / "requirements.txt"
    if req.exists():
        print(f"[依赖] 安装 {req}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=True)

# ─────────────────────────────────────────────
# 6. 主对话循环
# ─────────────────────────────────────────────
def reload_skills_into(messages: list) -> None:
    """重新加载技能并热更新 system prompt + TOOLS"""
    prompt_skills, tool_skills = load_skills()
    new_system = build_system_prompt(prompt_skills)
    if messages and messages[0].get("role") == "system":
        messages[0] = {"role": "system", "content": new_system}
    else:
        messages.insert(0, {"role": "system", "content": new_system})
    TOOLS[:] = list(TOOLS_BASE) + tool_skills
    print(f"[技能] 当前工具数：{len(TOOLS)}（基础 {len(TOOLS_BASE)} + 技能 {len(tool_skills)}）")

def chat_loop():
    print("\n" + "═" * 52)
    print("  🗂  本地文件助手  |  基于 qwen2.5:7b + ollama")
    print("  支持：读文件 / 写文件 / 列目录 / 删文件")
    print("  技能命令：/skills  /reload  /install <md|zip|git> <url>")
    print("  输入 exit 或 quit 退出")
    print("═" * 52 + "\n")

    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
    reload_skills_into(messages)

    while True:
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

        # ── 内置命令 ──
        if user_input.startswith("/"):
            try:
                if user_input == "/skills":
                    print("已加载工具：")
                    for t in TOOLS:
                        print(f"  - {t['function']['name']}: {t['function']['description'][:60]}")
                elif user_input == "/reload":
                    reload_skills_into(messages)
                elif user_input.startswith("/install "):
                    parts = user_input.split(maxsplit=2)
                    if len(parts) < 3:
                        print("用法：/install <md|zip|git> <url>")
                    else:
                        kind, url = parts[1], parts[2]
                        if kind == "md":
                            install_skill_from_md(url)
                        elif kind == "zip":
                            install_skill_from_zip(url)
                        elif kind == "git":
                            install_skill_from_git(url)
                        else:
                            print(f"未知安装类型：{kind}")
                            continue
                        reload_skills_into(messages)
                else:
                    print(f"未知命令：{user_input}")
            except Exception as e:
                print(f"[命令失败] {e}")
            continue

        # ── 正常对话 ──
        messages.append({"role": "user", "content": user_input})

        # 多轮工具调用循环：模型可能连续调用多次工具，直到给出最终自然语言回复
        for round_idx in range(MAX_TOOL_ROUNDS):
            try:
                resp = client.chat(
                    model="qwen2.5:7b",
                    messages=messages,
                    tools=TOOLS,
                    options={"num_ctx": NUM_CTX, "num_predict": 2048},
                )
                check_context_usage(resp, f"第{round_idx + 1}次请求")
            except Exception as e:
                print(f"[连接 ollama 失败] {e}\n请确认 ollama 已启动，且已拉取 qwen2.5:7b\n")
                # 回滚本轮 user 消息，回到外层等待用户重新输入
                messages.pop()
                break

            msg = resp.message

            # ── 工具调用分支 ──
            if msg.tool_calls:
                # 只追加一次 assistant（带 tool_calls），不要在循环开头预先追加
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,  # type: ignore
                })

                for call in msg.tool_calls:
                    fn_name = call.function.name
                    fn_args = call.function.arguments
                    print(f"\n  🔧 调用工具：{fn_name}")
                    print(f"     参数：{json.dumps(fn_args, ensure_ascii=False)}")
                    try:
                        result = execute_tool(fn_name, fn_args)  # type: ignore
                    except Exception as e:
                        result = f"[工具执行失败] {e}"
                    print(f"  📋 结果：{result}")
                    print(f"  📏 工具结果字符数：{len(result)}\n")

                    # 关键：补上 name，便于模型把结果与对应 tool_call 关联
                    messages.append({
                        "role": "tool",
                        "name": fn_name,
                        "content": result,
                    })
                # 继续下一轮，让模型基于工具结果继续推理
                continue

            # ── 普通回复分支：模型给出最终自然语言答复 ──
            content = msg.content or ""
            messages.append({"role": "assistant", "content": content})
            print(f"助手：{content}\n")
            break
        else:
            # for 正常跑完都没 break，说明工具轮数超限
            print(f"[警告] 工具调用轮数超过 {MAX_TOOL_ROUNDS}，已终止本轮对话。\n")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    chat_loop()
