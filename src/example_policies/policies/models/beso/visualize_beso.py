import matplotlib.pyplot as plt
import torch


def vis_absdiff(beso_model, batch1, batch2):

    # Try to unnormalize to raw scale; fall back gracefully
    try:
        raw1 = beso_model.unnormalize_inputs(batch1)
        raw1 = batch1
    except Exception:
        raw1 = batch1
    try:
        raw2 = beso_model.unnormalize_inputs(batch2)
        raw2 = batch2
    except Exception:
        raw2 = batch2

    keys = [
        "observation.images.rgb_static",
        "observation.images.rgb_right",
        "observation.images.rgb_left",
    ]

    for key in keys:
        img1 = raw1[key][0].to(torch.float32)  # (C,H,W)
        img2 = raw2[key][0, 0].to(torch.float32)  # (C,H,W)

        # Force into [0,1] for both encoder & imshow
        img1 = img1.clamp_(0, 1).cpu()
        img2 = img2.clamp_(0, 1).cpu()

        # Abs-diff (no colormap)
        diff = (img1 - img2).abs()
        diff_rgb = diff.mean(dim=0, keepdim=True).expand(3, -1, -1)

        # Saliency (already contrast-boosted by _saliency_on_image)
        sal1 = beso_model._saliency_on_image(img1.unsqueeze(0))
        sal2 = beso_model._saliency_on_image(img2.unsqueeze(0))

        overlay1 = beso_model._overlay_saliency(img1.unsqueeze(0), sal1, max_alpha=2)
        overlay2 = beso_model._overlay_saliency(img2.unsqueeze(0), sal2, max_alpha=2)

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        axes[0].imshow(img1.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title(f"{key} – Batch1")
        axes[1].imshow(img2.permute(1, 2, 0).cpu().numpy())
        axes[1].set_title(f"{key} – Batch2")
        axes[2].imshow(diff_rgb.permute(1, 2, 0).cpu().numpy())
        axes[2].set_title("Abs Diff")
        axes[3].imshow(overlay1)
        axes[3].set_title("Batch1 + Saliency (red)")
        axes[4].imshow(overlay2)
        axes[4].set_title("Batch2 + Saliency (red)")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


def plot_action_channels(beso_model, actions, batch_idx=0):
    """
    Plot each action channel in its own graph.

    Args:
        actions (torch.Tensor or np.ndarray): shape (B, T, C)
        batch_idx (int): which batch element to plot
    """
    # Move to numpy for plotting
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()

    act = actions[batch_idx]  # shape (T, C)
    T, C = act.shape
    time = range(T)

    for c in range(C):
        plt.figure(figsize=(10, 3))
        plt.plot(time, act[:, c])
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.title(f"Channel {c}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_modality_curves(
    beso_model, curves: dict, title: str = "Modality importance over time"
):
    S = len(next(iter(curves.values())))
    x = range(S)
    plt.figure(figsize=(10, 4))
    for k, y in curves.items():
        plt.plot(x, y, label=k)
    plt.xlabel("Observation step (t)")
    plt.ylabel("Relative importance")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
