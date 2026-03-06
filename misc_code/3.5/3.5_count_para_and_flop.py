"""
Count parameters and FLOPs for CausVid original (WanModel) and distilled (CausalWanModel).
Uses absolute paths only. Run from any directory; script switches to code dir for imports and checkpoint.
"""
import os
import sys

# Absolute paths only (no relative join)
CODE_DIR = "/home/ysunem/ys_26.2/2.27_CausVid/code"
CKPT_DIR = "/home/ysunem/ys_26.2/2.27_CausVid/code/wan_models/Wan2.1-T2V-1.3B"

# Ensure we can import causvid and checkpoint path is valid from code dir
sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)

import torch


def count_parameters(model):
    """Total parameter count (trainable + non-trainable)."""
    return sum(p.numel() for p in model.parameters())


def main():
    # -------------------------------------------------------------------------
    # 1. Load diffusion backbones only (no VAE, no text encoder)
    # -------------------------------------------------------------------------
    from causvid.models.wan.wan_base.modules.model import WanModel
    from causvid.models.wan.causal_model import CausalWanModel

    load_from_ckpt = os.path.isdir(CKPT_DIR)
    if load_from_ckpt:
        print("Loading original model (WanModel) from:", CKPT_DIR)
        original_model = WanModel.from_pretrained(CKPT_DIR)
        print("Loading distilled model (CausalWanModel) from:", CKPT_DIR)
        distilled_model = CausalWanModel.from_pretrained(CKPT_DIR)
    else:
        print("Checkpoint dir not found, building models from config (param count only):", CKPT_DIR)
        # config from wan_models/Wan2.1-T2V-1.3B/config.json (1.3B)
        config = dict(
            model_type="t2v",
            patch_size=(1, 2, 2),
            text_len=512,
            in_dim=16,
            dim=1536,
            ffn_dim=8960,
            freq_dim=256,
            text_dim=4096,
            out_dim=16,
            num_heads=12,
            num_layers=30,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
        )
        original_model = WanModel(**config)
        distilled_model = CausalWanModel(**config)
    original_model.eval()
    distilled_model.eval()

    # -------------------------------------------------------------------------
    # 2. Exact parameter count
    # -------------------------------------------------------------------------
    n_orig = count_parameters(original_model)
    n_dist = count_parameters(distilled_model)

    print("\n" + "=" * 60)
    print("PARAMETERS (diffusion backbone only)")
    print("=" * 60)
    print(f"  Original (WanModel):           {n_orig:>15,}  ({n_orig / 1e9:.4f} B)")
    print(f"  Distilled (CausalWanModel):   {n_dist:>15,}  ({n_dist / 1e9:.4f} B)")
    print(f"  Same architecture:            {n_orig == n_dist}")
    print()

    # -------------------------------------------------------------------------
    # 3. FLOPs with actual training/inference input shape and interface
    # -------------------------------------------------------------------------
    # From config: image_or_video_shape = [1, 21, 16, 60, 104] -> B, F, C, H, W
    B, F, C, H, W = 1, 21, 16, 60, 104
    text_len = 512
    text_dim = 4096
    # seq_len: max sequence length after patch embedding (F * (H/p_h) * (W/p_w)), patch_size=(1,2,2)
    frame_seq = (H // 2) * (W // 2)  # 30 * 52 = 1560 per frame
    seq_len = F * frame_seq  # 32760

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_model = original_model.to(device)
    distilled_model = distilled_model.to(device)

    # Model.forward(x, t, context, seq_len): x [B, C, F, H, W], t [B], context list of [text_len, text_dim]
    x_batch = torch.randn(B, C, F, H, W, device=device, dtype=torch.float32)
    t_batch = torch.tensor([500], device=device, dtype=torch.long)
    context_list = [torch.randn(text_len, text_dim, device=device, dtype=torch.float32) for _ in range(B)]

    try:
        from fvcore.nn import FlopCountAnalysis

        print("=" * 60)
        print("FLOPs (one forward pass, training mode)")
        print("=" * 60)

        with torch.no_grad():
            flops_orig = FlopCountAnalysis(
                original_model,
                (x_batch, t_batch, context_list, seq_len),
            )
            total_orig = flops_orig.total()
            print(f"  Original (WanModel):           {total_orig:>15,}  ({total_orig / 1e12:.4f} TFLOPs)")

            flops_dist = FlopCountAnalysis(
                distilled_model,
                (x_batch, t_batch, context_list, seq_len),
            )
            total_dist = flops_dist.total()
            print(f"  Distilled (CausalWanModel):   {total_dist:>15,}  ({total_dist / 1e12:.4f} TFLOPs)")
            print()
    except Exception as e:
        print("FLOPs (fvcore):")
        print("  FlopCountAnalysis failed (custom ops like flex_attention/flash_attention may be uncounted):")
        print(f"  {type(e).__name__}: {e}")
        print("  Install: pip install fvcore")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
