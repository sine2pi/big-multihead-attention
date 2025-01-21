Attentions scores are there and flowing but optional as both qk and relative positional bias have been rolled into the out tensor.
It's there in case someone needs it for something. these are experiments. Otherwise, the out is your standard shape [batch, seq_length, dims] tensor. 
Just be aware of the bias. It changes and adjusts to loss. The max can be set when you initialize your model via config hyperparameter nax_dist. 
Or leave it be, eventually it will find the right spot for itself. I would still keep an eyeball on it though. The changes in bias print to screen while training.
Max_dist is your starting point for bias with a 1:1 token ratio. Deepending on your use case you may want to remove the positional scaling.


    
    
    class MultiheadAttention(nn.Module):
        def __init__(self, base, n_state, n_head, max_dist):
            super().__init__()
            self.base = base
            self.n_state = n_state
            self.n_head = n_head
            self.max_dist = max_dist
    
            assert self.n_state % self.n_head == 0, "n_state must be divisible by n_head"
            self.h_dim = self.n_state // self.n_head
            assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"
    
            self.query = nn.Linear(self.n_state, self.n_state)
            self.key = nn.Linear(self.n_state, self.n_state, bias=False)
            self.value = nn.Linear(self.n_state, self.n_state)
            self.out = nn.Linear(self.n_state, self.n_state)
    
            self.combined_rotary = CombinedRotaryEmbedding(base, n_state, n_head)
            self.kv_cache = {}
            
            self.positional_scaling = nn.Parameter(torch.ones(1))
            self.rel_pos_bias = nn.Parameter(torch.zeros((2 * int(self.max_dist) - 1, self.n_head))).to(device)
            self.inv_freq = nn.Parameter(data=1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim)))
    
        def update_base(self, new_base):
            if new_base is not None and new_base != self.base:
                self.base = new_base
                inv_freq = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
                self.inv_freq.data.copy_(inv_freq)
                self.combined_rotary.update_base(self.base)
    
        def update_dist(self, new_dist):
            if new_dist is not None and new_dist != new_dist:
                self.max_dist = new_dist
                rel_pos_bias = nn.Parameter(torch.zeros((2 * int(self.max_dist) - 1, self.n_head)))
                self.rel_pos_bias.data.copy_(rel_pos_bias)
            
        def forward(self, x: Tensor, xa: Optional[Tensor] = None, 
                    mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None, new_dist=None, new_base=None):
            
            self.update_base(new_base) 
            self.update_dist(new_dist)
    
            q = self.query(x)
    
            if kv_cache is None or xa is None or self.key not in kv_cache:
                k = self.key(x if xa is None else xa)
                v = self.value(x if xa is None else xa)
            else:
    
                k = kv_cache[self.key]
                v = kv_cache[self.value]
    
            q = self.combined_rotary(q) * self.positional_scaling
            k = self.combined_rotary(k) * self.positional_scaling
    
            wv, qk = self.qkv_attention(q, k, v, mask)
            return self.out(wv), qk
       
        def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tuple     [torch.Tensor, Optional[torch.Tensor]]:
            n_batch, n_ctx, n_state = q.shape
            scale = (n_state // self.n_head) ** -0.25
            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    
            seq_len_q = q.size(2)
            seq_len_k = k.size(2)
            positions = torch.arange(end=seq_len_q, device=q.device).unsqueeze(dim=1) - torch.arange(end=seq_len_k, device=q.device).unsqueeze(dim=0)
            positions = positions.clamp(min=-self.max_dist + 1, max=self.max_dist - 1) + self.max_dist - 1
            rel_bias = self.rel_pos_bias[positions]
            rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)  
    
            
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale 
            attn_scores = attn_scores + rel_bias
    
            a = scaled_dot_product_attention(q, k, v, attn_mask = attn_scores , is_causal=mask is not None and n_ctx > 1) 
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            attn_scores = attn_scores.reshape(n_batch, n_ctx, -1)[:,:,:self.n_state]
    
            return out, attn_scores
