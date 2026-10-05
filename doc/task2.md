# 任务2：补全代码

核心想法是在生成 qkv 的时候不需要每次重新生成，而是利用之前的缓存内容。

## 数据结构

一个 `kv_cache` 列表，每个 Transformer block（层）对应一个字典，`k/v` 初始化为 `None`，表示这一层还没有缓存任何 `key/value`：

```py
def new_kv_cache(params):
    # 新建一个空的 KV cache：每层一个 {'k': None, 'v': None}
    return [{'k': None, 'v': None} for _ in params['blocks']]
```

每经过一次 `generate`，`kv_cache` 的相应字段就会更新，包含之前所有生成结果，这样下一次只需要再加入下一个输入 token 即可。

## 函数修改

### `generate`

生成 `kv_cache`，第一步喂完整的 prompt，之后只需喂最新 token（即 `[next_id]`）。

### `gpt2`

输入的下标要改为从 `n_past` 开始，其中 `n_past` 根据 `kv_cache` 获取。每层独立缓存。

### `mha`

把历史 token 的 `k/v` 拼到当前步前面，获取 `qkv` 并将当前结果写回缓存。

