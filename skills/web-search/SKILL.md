---
name: web-search
type: tool
description: 执行网络搜索
entry: handler.py:run
parameters:
  - name: text
    type: string
    description: 要网络检索的文本
    required: true
---

调试用工具，仅在用户明确说"网络检索"或"搜索"时调用。
