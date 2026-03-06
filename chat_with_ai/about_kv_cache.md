
下面是 CausVid 里 KV cache 的用法和实现，按「初始化 → 推理入口 → 模型内部 → 单层 attention」整理，方便你对照学习。

---

## 1. 接口定义（`model_interface.py`）

```14:27:code/causvid/models/model_interface.py
    def forward(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None
    ) -> torch.Tensor:
        """
        ...
            - kv_cache: a list of dictionaries containing the key and value tensors for each attention layer.
            - current_start: the start index of the current frame in the sequence.
            - current_end: the end index of the current frame in the sequence.
```

`kv_cache` 是「每层一个 dict」的列表，每个 dict 里是当前层的 key/value；`current_start` / `current_end` 表示当前块在整段序列里的 token 范围。

---

## 2. 初始化 KV cache（`causal_inference.py`）

```47:59:code/causvid/models/wan/causal_inference.py
    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, 32760, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 32760, 12, 128], dtype=dtype, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache
```

- 每个 block 一个 dict：`"k"` / `"v"`，shape 为 `[B, seq_len, num_heads, head_dim]`（这里 32760≈整段视频 token 数，12 heads，128 head_dim）。
- 推理前若 cache 为空会先调这个初始化（见下面推理循环前的 `_initialize_kv_cache` 调用）。

---

## 3. 推理时如何使用（`causal_inference.py`）

- 第一次进入时初始化 cache 和 crossattn_cache；之后只重置 crossattn 的 `is_init`，KV cache 复用。
- 每个 **block**（时间块）用固定的 `current_start` / `current_end`，和同一个 `kv_cache1` 传给 generator。

```99:114:code/causvid/models/wan/causal_inference.py
        # Step 1: Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            ...
```

```133:142:code/causvid/models/wan/causal_inference.py
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                    current_end=(block_index + 1) *
                    self.num_frame_per_block * self.frame_seq_length
                )
```

也就是说：**按时间块递进**，每块只算当前块的 token，用 `current_start:current_end` 往 cache 里写/读。

---

## 4. Wrapper 把 kv_cache 传给 model（`wan_wrapper.py`）

```164:194:code/causvid/models/wan/wan_wrapper.py
    def forward(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None
    ) -> torch.Tensor:
        ...
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(...)  # 无 kv_cache 时走普通 forward
```

有 `kv_cache` 时多传 `current_start` / `current_end`，并走带 cache 的推理分支。

---

## 5. 主模型：根据是否有 kv_cache 分支（`causal_model.py`）

```693:701:code/causvid/models/wan/causal_model.py
    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)
```

- **有 kv_cache** → `_forward_inference`（逐块、写/读 cache）。
- **无 kv_cache** → `_forward_train`（正常训练 forward）。

---

## 6. Inference 时每层传入自己的 cache 槽位（`causal_model.py`）

```556:568:code/causvid/models/wan/causal_model.py
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                assert False
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "current_end": current_end
                    }
                )
                x = block(x, **kwargs)
```

每一层用 `kv_cache[block_index]`，即「该层对应的一个 `{"k", "v"}` dict」，并带上统一的 `current_start` / `current_end`。

---

## 7. 单层 Self-Attention 的 KV cache 读写（`causal_model.py`）

这是 **真正读写 KV cache** 的地方：无 cache 时用 flex_attention；有 cache 时只算当前块的 q，把当前块的 k/v 写入 cache，再用「从 0 到 current_end 的 k/v」做 attention。

```86:152:code/causvid/models/wan/causal_model.py
    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, current_end=0):
        ...
        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # 训练 / 无 cache：全序列 RoPE + flex_attention
            roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
            roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)
            # ... padding ...
            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, :-padded_length].transpose(2, 1)
        else:
            # 推理 + KV cache：只对当前块做 RoPE，写入 cache，再 attention
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start // math.prod(grid_sizes[0][1:]).item()).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start // math.prod(grid_sizes[0][1:]).item()).type_as(v)

            kv_cache["k"][:, current_start:current_end] = roped_key
            kv_cache["v"][:, current_start:current_end] = v

            x = attention(roped_query, kv_cache["k"][:, :current_end], kv_cache["v"][:, :current_end])
```

要点：

- **写**：当前块的 `roped_key`、`v` 写入 `kv_cache["k"]` / `kv_cache["v"]` 的 `[current_start:current_end]`。
- **读**：attention 时用的是「从 0 到 current_end 的整段历史」：`kv_cache["k"][:, :current_end]`、`kv_cache["v"][:, :current_end]`，实现因果的「只看过去和当前」。

---

## 8. 使用的 attention 实现（`wan_base/modules/attention.py`）

```131:175:code/causvid/models/wan/wan_base/modules/attention.py
def attention(
    q,
    k,
    v,
    ...
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(...)
    else:
        ...
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)
```

有 cache 时，这里收到的 `k/v` 已经是「从 cache 里切出来的 `[:, :current_end]`」，所以不需要再在 attention 里区分 cache，只要保证 `q` 是当前块、`k/v` 是历史+当前即可。

---

## 小结（便于你模仿）

| 环节 | 作用 |
|------|------|
| **结构** | `kv_cache = [{"k": Tensor, "v": Tensor}, ...]`，一层一个 dict，shape `[B, max_seq, num_heads, head_dim]`。 |
| **初始化** | 按 block 数预分配全零 `k/v`，长度覆盖整段序列。 |
| **推理循环** | 按时间块递增 `current_start` / `current_end`，同一份 cache 跨 step 复用。 |
| **写 cache** | 当前块算出的 `roped_key`、`v` 写入 `kv_cache["k"][:, current_start:current_end]` 等位置。 |
| **读 cache** | attention 时用 `kv_cache["k"][:, :current_end]` 和 `kv_cache["v"][:, :current_end]`，实现因果 + 复用历史 K/V。 |

如果你愿意，我可以再根据你当前项目（例如 3.3_diffsynth 里的某个模型）画一版「该怎么加同样结构的 KV cache」的伪代码或接口设计，方便你直接对照改。