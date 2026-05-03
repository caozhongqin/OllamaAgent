from pathlib import Path
HERE = Path(__file__).parent

def run(args: dict) -> str:
    return f"[检索结果] Hello, {args['text']}!"
