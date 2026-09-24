"""Gated DeltaNet（Qwen3-Next / Kimi Linear 一系的线性注意力）。

**核心思想**：用**定长**循环状态 ``S ∈ R^{d_k × d_v}`` 取代 KV cache，复杂度从
O(T²) 降到 O(T)。递推（每个头独立）::

    alpha_t = sigmoid(W_alpha x_t)                                   # 逐头衰减门（标量）
    beta_t  = sigmoid(W_beta  x_t)                                   # 写入强度
    S_t = alpha_t * S_{t-1} + beta_t * k_t (v_t - alpha_t * S_{t-1}^T k_t)^T
    o_t = S_t^T q_t

即「先按 alpha 衰减旧状态，再用 delta 规则写入新信息」：Mamba-2 风格的门控 +
DeltaNet 的 delta 更新（写入的是 ``v_t`` 与「当前状态预测值」之差，而非直接写入
``v_t``；注意括号里是**衰减之后**的 ``alpha_t * S_{t-1}``，这样等价于
``S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T``）。

**缓存即状态**：状态矩阵的大小 ``B*H*d_k*d_v`` 与已处理 token 数**无关**（这正是
它区别于 KV cache 的本质）。代价是：状态里**没有序列维**，所以「已处理多少个 token」
这个信息必须**显式带在状态里**。本实现的状态是 ``(S, n)``（``S`` 是定长矩阵，
``n`` 是已处理的 token 数），``state_seq_len`` 返回 ``n``。

之所以要返回真实计数而不是 0：``arch/model.py`` 里 ``state_seq_len`` 承担两个职责 ——
``ArchModel.forward`` 用它算位置编码的 ``start_pos``，``generate`` 用它切片
``input_ids[:, past_len:]``。对线性注意力两者要的都是「已处理多少 token」；
若返回 0，``generate`` 每步会重喂整段 ``input_ids``，而这些 token 会被**再次叠加**
进已有状态，造成重复计数。

两条前向路径（数值上必须一致，见 ``_chunked_forward`` / ``_recurrent_forward``）:

- 训练（``past_key_value is None``）：chunked 并行算法，块内全部是矩阵运算，
  块间串行传递状态；复杂度 O(T·C·d) 而不是逐 token 的 Python 循环。
- 推理（``past_key_value is not None``）：逐 token 递推，只更新定长状态。

位置编码：Gated DeltaNet 是因果递推结构，位置信息由状态累积隐式携带（Qwen3-Next
等混合模型也只在全注意力层用 RoPE）。``positional`` 仅为接口一致性保留，不参与计算。

关于门控形式：这里实现**逐头标量** alpha（配置里也可换成别的形式），因为
chunked 并行算法中「衰减」只有作为标量时才能从状态转移里提出去、写成
``S_t = gamma_t * Phi_t S_0 + Σ ...`` 的形式（见下方推导）。Qwen3-Next 用的是
逐通道衰减，其 chunked 形式需要额外引入非对称 rank-1 更新，收敛性与实现复杂度
都更高，本实现按上面的规格取标量。
"""
import torch
import torch.nn.functional as F
from torch import nn

from ..registry import register

# 归一化的最小分母，避免除零；与参考实现（fla / Qwen3-Next）的 1e-6 一致
_NORM_EPS = 1e-6
# 衰减门的 log 下限：alpha 可能非常接近 0，log 后是很小的负数，clamp 只为数值安全
_LOG_EPS = 1e-12


@register("attention", "deltanet")
class GatedDeltaNet(nn.Module):
    def __init__(self, cfg, positional=None):
        super().__init__()
        hidden = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        # 状态是 d_k × d_v 的方阵，这里取 d_k = d_v = head_dim
        self.head_dim = cfg.head_dim
        # chunk 的长度：块内并行、块间串行。64 是 Qwen3-Next / fla 的常用值
        self.chunk_size = int(cfg.get("chunk_size", 64))

        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        # 门控投影：alpha（衰减）、beta（写入强度），逐头标量
        self.alpha_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.beta_proj = nn.Linear(hidden, self.num_heads, bias=False)

        # q/k 做 L2 归一化：参考实现（fla 的 use_qk_l2norm_in_kernel、Qwen3-Next）默认开启。
        # 它同时是数值稳定性的关键 —— delta 规则里要解 (I + tril(beta * K K^T))^{-1}，
        # 未归一化的 k 会让该矩阵条件数爆炸，chunked 与递推两条路径就会对不上。
        self.qk_l2norm = bool(cfg.get("qk_l2norm", True))
        # 位置编码：递推结构自带因果位置信息，这里不使用（保留参数以统一接口）
        self.positional = positional

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout

    # ------------------------------------------------------------------ #
    @staticmethod
    def state_seq_len(state) -> int:
        """返回状态里**已处理的 token 数**（存在状态元组的第 1 项里）。

        线性注意力的状态矩阵本身没有序列维，所以这个计数必须显式携带：
        模型侧既要用它算位置编码的 ``start_pos``，也要用它在 ``generate``
        里切片 ``input_ids[:, past_len:]``（只喂新 token，避免把历史重复计入状态）。
        """
        if state is None:
            return 0
        # 兼容只有状态矩阵、没有计数的调用方（视为「尚未处理任何 token」）
        return int(state[1]) if len(state) > 1 else 0

    # ------------------------------------------------------------------ #
    def forward(self, x, position_embeddings=None, past_key_value=None, use_cache=False, attention_mask=None):
        """``attention_mask`` 对本组件无用：递推天然因果。若要屏蔽 padding 位置的
        token，需要把该位置的 ``beta`` 置 0（本实现未做，训练数据按定长序列打包）。
        """
        bsz, seq_len, _ = x.shape
        dtype = x.dtype
        # 投影按模型 dtype 算（与其它组件一致）；下面的递推/chunked 运算内部升到 fp32：
        # 要解一个 C×C 的下三角线性系统，bf16 精度不够（fla 的 kernel 同样在 fp32 里解）
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        # 门控在 fp32 里算：sigmoid 后要取 log 累积，bf16 下接近 1 的门会丢精度
        alpha = torch.sigmoid(self.alpha_proj(x).float())   # [B, T, H]
        beta = torch.sigmoid(self.beta_proj(x).float())     # [B, T, H]
        q, k, v = q.float(), k.float(), v.float()
        if self.qk_l2norm:
            q = q / q.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS)
            k = k / k.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS)

        if past_key_value is None:
            # 训练 / 首次前向：chunked 并行
            past_len = 0
            output, state = self._chunked_forward(q, k, v, alpha, beta)
        else:
            # 推理：逐 token 递推（T 通常为 1），从已有定长状态继续。
            # 状态是 (S, n)：S 是定长矩阵，n 是 S 里已经"消化"掉的 token 数
            past_len = int(past_key_value[1]) if len(past_key_value) > 1 else 0
            output, state = self._recurrent_forward(q, k, v, alpha, beta, past_key_value[0])

        # 计数随前向推进：模型侧靠它切片 input_ids / 算 start_pos
        present_state = (state, past_len + seq_len) if use_cache else None
        output = output.reshape(bsz, seq_len, -1).to(dtype)
        output = self.resid_dropout(self.o_proj(output))
        return output, present_state

    # ------------------------------------------------------------------ #
    def _recurrent_forward(self, q, k, v, alpha, beta, state):
        """逐 token 递推（推理路径）。``state``: [B, H, d_k, d_v] 的 fp32 定长状态。

        直接照抄定义::

            S_t = alpha_t * S_{t-1} + beta_t * k_t (v_t - alpha_t * S_{t-1}^T k_t)^T
            o_t = S_t^T q_t
        """
        _, seq_len, _, _ = q.shape
        outputs = []
        for t in range(seq_len):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]           # [B, H, d]
            at = alpha[:, t].unsqueeze(-1).unsqueeze(-1)      # [B, H, 1, 1]
            bt = beta[:, t].unsqueeze(-1).unsqueeze(-1)
            decayed = at * state                              # alpha_t * S_{t-1}
            # (v_t - alpha_t S_{t-1}^T k_t) -> [B, H, d_v]
            delta = vt - (decayed.transpose(-1, -2) @ kt.unsqueeze(-1)).squeeze(-1)
            state = decayed + bt * (kt.unsqueeze(-1) * delta.unsqueeze(-2))
            outputs.append((state.transpose(-1, -2) @ qt.unsqueeze(-1)).squeeze(-1))  # S_t^T q_t
        return torch.stack(outputs, dim=1), state

    # ------------------------------------------------------------------ #
    def _chunked_forward(self, q, k, v, alpha, beta):
        """chunked 并行算法（训练路径）。

        推导（单头，块内局部下标 ``1..C``，进入本块前的状态记为 ``S_0``）：
        把递推写成 ``S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T``，
        展开得::

            S_t = gamma_t * Phi_t S_0 + Σ_i (gamma_t / gamma_i) Phi_i^t beta_i k_i v_i^T

        其中 ``gamma_t = Π_{j<=t} alpha_j`` 是块内累积衰减，``Phi_i^t = Π_{j=i+1..t}
        (I - beta_j k_j k_j^T)`` 是**不含衰减**的状态转移。用 delta 规则的 WY 表示
        （``Phi_i^t k_i = k_i + Σ_{j>i} T[j,i] k_j``，``T = (I + L)^{-1}``，
        ``L[j,i] = beta_j k_j·k_i``，严格下三角）把双重求和整理成矩阵形式，得到::

            U   = T diag(beta ⊙ exp(g_C - g)) V                  # 「修正后的 value」
            W   = T diag(beta) K                                 # 状态转移的写回向量
            Phi = tril(Q K^T)                                    # 块内注意力（含对角）
            E[t,i] = exp(g_t - g_i)  (i<=t)                      # 块内两两衰减，恒 <= 1
            o_intra = ((Phi T) ⊙ E ⊙ beta) V
            o_inter = exp(g) ⊙ (Q - Phi W) S_0                   # 来自历史状态的贡献
            S_new   = exp(g_C) (S_0 - K^T W S_0) + K^T U         # 传给下一块的状态

        其中 ``g = cumsum(log alpha) <= 0``。所有衰减因子都以 ``exp(g_t - g_i)`` 或
        ``exp(g)`` 的形式出现，恒 <= 1，不会溢出（强衰减时下溢到 0 是物理上正确的）。
        块内只有矩阵乘法，块间串行数量是 ``T / C``，与逐 token 循环有本质区别。

        ``C`` 之外的填充位置取 ``beta = 0, alpha = 1``：既不写入状态、也不产生衰减，
        等价于不存在，最后把输出切掉即可。
        """
        bsz, seq_len, heads, dim = q.shape
        device, dtype = q.device, q.dtype
        C = min(self.chunk_size, max(seq_len, 1))
        pad = (-seq_len) % C
        n = seq_len + pad

        def _prep(t, pad_value):
            """[B, T, H, ...] -> [B, H, n/C, C, ...]（按需在序列尾补齐）。"""
            if pad:
                # F.pad 的元组从最后一维往前数：把 dim=1（序列维）补到 n
                t = F.pad(t, (0,) * (2 * (t.dim() - 2)) + (0, pad), value=pad_value)
            return t.transpose(1, 2).reshape(bsz, heads, n // C, C, *t.shape[3:])

        q_c = _prep(q, 0.0)
        k_c = _prep(k, 0.0)
        v_c = _prep(v, 0.0)
        alpha_c = _prep(alpha, 1.0)               # 填充位置不衰减
        beta_c = _prep(beta, 0.0)                 # 填充位置不写入

        # 块内因果掩码
        causal = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device))
        strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=-1)
        eye = torch.eye(C, dtype=dtype, device=device)

        state = torch.zeros(bsz, heads, dim, v.shape[-1], dtype=dtype, device=device)
        outputs = []
        for c in range(n // C):
            K, Q, V = k_c[:, :, c], q_c[:, :, c], v_c[:, :, c]        # [B, H, C, d]
            b = beta_c[:, :, c]                                        # [B, H, C]
            g = torch.cumsum(torch.log(alpha_c[:, :, c].clamp_min(_LOG_EPS)), dim=-1)  # <= 0
            g_last = g[..., -1].unsqueeze(-1)                          # [B, H, 1]

            # T = (I + L)^{-1}，L[j,i] = beta_j * (k_j · k_i)，严格下三角（beta 取行下标）
            #
            # 在 fp32 下求解：I + L 是单位下三角，行列式恒为 1、必然可逆，但条件数
            # 可能很大（k 已 L2 归一化，|L| <= 1，逆的幅值随 chunk 长度指数增长），
            # 多给几位有效数字是划算的 —— 这里是 O(C^3) 的小矩阵，代价可忽略。
            L = torch.tril((b.unsqueeze(-1) * (K @ K.transpose(-1, -2))), diagonal=-1)
            A = torch.eye(C, dtype=torch.float32, device=device) + L.float()
            T = torch.linalg.solve_triangular(
                A, torch.eye(C, dtype=torch.float32, device=device).expand(bsz, heads, C, C),
                upper=False, unitriangular=True,
            ).to(dtype)

            # 块内写入权重：beta_m * exp(g_C - g_m)（恒 <= beta_m）
            write = b * torch.exp(g_last - g)
            U = T @ (write.unsqueeze(-1) * V)                          # [B, H, C, d_v]
            W = T @ (b.unsqueeze(-1) * K)                              # [B, H, C, d_k]

            Phi = torch.tril(Q @ K.transpose(-1, -2))                  # 块内注意力，含对角
            # 块内两两衰减 E[t,i] = exp(g_t - g_i)（i <= t），恒 <= 1。
            #
            # ⚠️ 未来位置（i > t）的 g_t - g_i > 0，块内累积衰减很大时（实测真实数据 +
            # 56% padding 时单块可达 -123）exp 会**上溢成 inf**；此时再乘因果掩码就是
            # inf * 0 = NaN，整条链路随之中毒。所以先把指数夹到 <= 0（这些位置本来就要被
            # 掩掉，取值无物理意义），再乘掩码 —— 严格下三角部分不受影响，因此
            # chunked 与逐 token 递推的等价性不变。
            decay = torch.exp((g.unsqueeze(-1) - g.unsqueeze(-2)).clamp_max(0.0)) * causal
            o_intra = ((Phi @ T) * decay * b.unsqueeze(-2)) @ V
            # 历史状态的贡献：先做 (Q - Phi W) 的「修正 query」，再乘以衰减
            o_inter = torch.exp(g).unsqueeze(-1) * ((Q - Phi @ W) @ state)
            outputs.append(o_intra + o_inter)

            # 块间状态传递：衰减 + delta 修正 + 新写入
            state = torch.exp(g_last).unsqueeze(-1) * (state - K.transpose(-1, -2) @ (W @ state)) \
                + K.transpose(-1, -2) @ U

        output = torch.cat(outputs, dim=2)                             # [B, H, n/C, C, d_v]
        output = output.reshape(bsz, heads, n, -1)[:, :, :seq_len]     # 去掉填充
        return output.transpose(1, 2), state                            # [B, T, H, d_v]
