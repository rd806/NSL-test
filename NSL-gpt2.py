import numpy as np
import torch
import time
import math
torch.set_printoptions(8)

def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]
        
        Input: Tensor
        Output: Tensor
    """
    # x**3 一般比torch.pow(x, 3)更快
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    # subtract the max for numerical stability (does not change the result)
    # 对所有数值都减去它们当中的最大值
    x = x - x.max(dim=-1, keepdim=True).values
    # 进行指数放大
    e = torch.exp(x)
    # 输出向量
    return e / e.sum(dim=-1, keepdim=True)


def layer_norm(x, g_b, eps:float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input: 
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """
    # g 存储 gamma，b 存储 beta
    # 获取这两个值
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])

    # normalize over the last (embedding) dimension, then scale and shift
    # 求出均值
    mean = x.mean(dim=-1, keepdim=True)
    # 求出方差
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    # 标准化（分母加上eps防止发生除零运算）
    x_hat = (x - mean) / torch.sqrt(var + eps)
    # 恢复 gamma 和 beta
    return x_hat * g + b

def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer 
        Input: 
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    # keep everything as torch tensors so that downstream ops (e.g. `.chunk`) work
    # 获取张量 x，w，b
    x = torch.as_tensor(x)
    w = torch.as_tensor(w_b['w']).to(x.dtype)
    b = torch.as_tensor(w_b['b']).to(x.dtype)
    # 直接进行矩阵运算
    return x @ w + b
    

def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input: 
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']
    # 复用实现的 linear 和 gelu
    return linear(gelu(linear(x, w_b1)), w_b2)


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input: 
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    # 获取 k 的维度，直接读取张量最后一维的值
    d_k = k.shape[-1]
    # scaled dot-product attention: softmax(q k^T / sqrt(d_k) + mask) v
    # 计算权重
    # k.transpose(-2, -1) 直接执行转置运算
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    scores = scores + mask
    weights = softmax(scores)
    return weights @ v

def mha(x, attn, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention
        
        Input: 
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    # qkv projection
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    
    # Split into qkv
    """
        Task: Split the q,k,v matrix from the tensor x
        Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
    """

    # 从张量 x 中得到 qkv
    """
        chunk 函数调用：x.chunk(chunks, dim)
            - chunks：期望返回的块数量（整数）。
            - dim：沿哪个维度进行拆分，默认为 0。
    """
    # 按最后一个维度均匀拆分
    qkv = x.chunk(3, dim=-1)  # 3 * [n_seq, n_embd]

    # Split into heads
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
    qkv_heads = list(zip(*qkv_heads))  # [3, n_head, n_seq, n_embd/n_head]

    # Causal mask to hide future inputs from being attended to
    """
        Task: Construct mask matrix
        Notes: 
            | 0  -inf -inf ... -inf |
            | 0    0  -inf ... -inf |
            | 0    0    0  ... -inf |
            |...  ...  ... ...  ... | 
            | 0    0    0  ...   0  |
        Mask is a tensor whose dimension is [n_seq, n_seq]
    """
    # 获取张量第0维的大小
    n_seq = x.shape[0]
    # 构建 mask 矩阵
    causal_mask = torch.triu(
        # 创建一个全为 -inf 的矩阵
        torch.full((n_seq, n_seq), float('-inf'), dtype=x.dtype), 
        # 保留主对角线往上偏移 1 格的右上角范围内的元素，其余置零
        diagonal=1
    )

    # Perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in qkv_heads]  # n_head * [n_seq, n_embd/n_head]
    
    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    # 张量拼接
    x = torch.cat(out_heads, dim=-1)  # [n_seq, n_embd]

    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    return x


def transformer_block(x, block, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    
    x = torch.Tensor(x)
    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids

def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target, n_tokens_to_generate, K):
    
    """
        Task: Load 124M and 1558M models at the same time, use greedy sampling, and complete speculative decoding
    
        Inputs:
            inputs (list): The initial list of token IDs from the prompt.
            draft_params, target_params: Model weights for the draft and target models.
            hparams_draft, hparams_target: Hyperparameters for both models.
            n_tokens_to_generate (int): The number of new tokens to generate.
            K (int): The number of tokens the draft model speculates at each step (e.g., 4).

        Returns:
            list: A list of newly generated token IDs.
            
    """
    generated_ids = []
    current_inputs = list(inputs)

    while len(generated_ids) < n_tokens_to_generate:
        # 1. the draft model greedily proposes K tokens, one autoregressive step each
        draft_ids = []
        for _ in range(K):
            draft_logits = gpt2(current_inputs + draft_ids, draft_params, hparams_draft["n_head"])
            draft_ids.append(int(np.argmax(draft_logits[-1])))

        # 2. the target model scores the whole prompt + K drafted tokens in ONE forward pass
        target_logits = gpt2(current_inputs + draft_ids, target_params, hparams_target["n_head"])
        n = len(current_inputs)

        # target_logits[n - 1 + i] is the target's greedy prediction for the token
        # following position n - 1 + i, i.e. its choice for draft_ids[i].
        # Accept the longest prefix on which draft and target agree.
        accepted = []
        for i in range(K):
            target_id = int(np.argmax(target_logits[n - 1 + i]))
            if target_id == draft_ids[i]:
                accepted.append(draft_ids[i])
            else:
                # first mismatch: this token is corrected by the target, then we stop
                accepted.append(target_id)
                break
        else:
            # all K drafts were accepted -> the target's next token comes for free
            accepted.append(int(np.argmax(target_logits[n - 1 + K])))

        # 3. commit the accepted tokens, truncating any overshoot on the last round
        accepted = accepted[: n_tokens_to_generate - len(generated_ids)]
        generated_ids.extend(accepted)
        current_inputs.extend(accepted)

    return generated_ids


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    start = time.time()
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)