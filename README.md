class RelativePositionalBias(nn.Module):
    def __init__(self, max_dist: int, n_head: int):
        super().__init__()
        self.max_dist = max_dist
        self.n_head = n_head
        self.pos_bias = nn.Parameter(torch.zeros((2 * int(self.max_dist) - 1, self.n_head)))

    def forward(self, seq_len_q, seq_len_k, device):
        positions = torch.arange(end=seq_len_q, device=device).unsqueeze(dim=1) - torch.arange(end=seq_len_k, device=device).unsqueeze(dim=0)
        positions = positions.clamp(min=-self.max_dist + 1, max=self.max_dist - 1) + self.max_dist - 1
        rel_bias = self.pos_bias[positions]
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_head, seq_len_q, seq_len_k)
        return rel_bias

    def update_dist(self, new_dist, device):
        if new_dist is not None and new_dist != self.max_dist:
            self.max_dist = new_dist
            self.pos_bias = nn.Parameter(torch.zeros((2 * int(self.max_dist) - 1, self.n_head))).to(device)

class MultiheadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int, n_freq: int):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head

        assert self.n_state % self.n_head == 0, "n_state must be divisible by n_head"
        self.h_dim = self.n_state // self.n_head
        assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"

        self.query = nn.Linear(self.n_state, self.n_state)
        self.key = nn.Linear(self.n_state, self.n_state, bias=False)
        self.value = nn.Linear(self.n_state, self.n_state)
        self.out = nn.Linear(self.n_state, self.n_state)
        self.givens_rotary = CombinedRotaryEmbedding(n_state, n_head, n_freq)
        self.kv_cache = {}

        self.positional_scaling = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, xa: Optional[Tensor] = None,
                mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None,
                rel_pos_bias: Optional[Tensor] = None, temperature: float = 1.0,
                span_scale: float = 1.0) -> tuple[Any, Tensor | None]:

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        q = self.givens_rotary(q)
        k = self.givens_rotary(k)
        wv, qk = self.qkv_attention(q, k, v, mask, rel_pos_bias, temperature, span_scale)
        return self.out(wv), qk

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None,
                      rel_pos_bias: Optional[Tensor] = None, temperature: float = 1.0,
                      span_scale: float = 1.0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        if rel_pos_bias is not None:
            attn_scores = attn_scores + rel_pos_bias
            
        attn_scores = attn_scores / temperature 
        attn_scores = attn_scores * (1.0 / span_scale) 
        attn_scores = attn_scores * scale 

        if MultiheadAttention.use_sdpa:
            a = scaled_dot_product_attention(q, k, v, attn_mask=attn_scores, is_causal=mask is not None and n_ctx > 1)
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2) 
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk
