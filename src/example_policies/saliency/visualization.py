"""Visualization functions for saliency maps."""

import logging

import matplotlib.pyplot as plt
import numpy as np


def visualize_saliency(
    images_dict,
    image_saliency_dict,
    state_saliency,
    output_path,
    episode_idx,
    frame_idx,
):
    """
    Visualize original images with overlaid saliency maps and state saliency.

    Creates a multi-panel figure showing:
    - Original images (left column)
    - Images with saliency heatmap overlays (right column)
    - State vector saliency bar chart (bottom row, if state exists)
    - Image vs State comparison pie chart (bottom right, if state exists)

    Args:
        images_dict: Dict of image tensors {key: [B, C, H, W]}
        image_saliency_dict: Dict of saliency maps {key: [B, H, W]}
        state_saliency: State saliency tensor [B, D] or None
        output_path: Path to save the figure
        episode_idx: Episode index for title
        frame_idx: Frame index for title
    """
    image_keys = sorted(images_dict.keys())
    n_images = len(image_keys)

    # Calculate saliency magnitudes using top 5% to avoid background dilution
    def top_k_mean(tensor, k=0.05):
        """Calculate mean of top k% values."""
        flat = tensor.flatten()
        k_count = max(1, int(len(flat) * k))
        topk_values = flat.topk(k_count).values
        return topk_values.mean().item()

    total_image_saliency = sum(top_k_mean(sal) for sal in image_saliency_dict.values())
    total_state_saliency = (
        top_k_mean(state_saliency) if state_saliency is not None else 0.0
    )

    # Setup figure layout
    has_state = state_saliency is not None
    n_rows = n_images + (2 if has_state else 0)  # Extra row for top-n comparison
    fig = plt.figure(figsize=(12, 4 * n_rows))
    gs = (
        fig.add_gridspec(n_rows, 3, width_ratios=[1, 1, 1])
        if has_state
        else fig.add_gridspec(n_rows, 2)
    )

    # Plot all image saliency maps
    for idx, key in enumerate(image_keys):
        img = images_dict[key][0].cpu()
        saliency = image_saliency_dict[key][0].numpy()

        # Convert image to displayable format
        if img.shape[0] == 3:  # RGB
            img_np = np.clip(img.permute(1, 2, 0).numpy(), 0, 1)
        else:  # Grayscale or depth
            img_np = img[0].numpy()

        # Plot original and saliency overlay
        for col, (title_suffix, show_saliency) in enumerate(
            [("Original", False), ("Saliency", True)]
        ):
            ax = fig.add_subplot(gs[idx, col])
            ax.imshow(img_np, cmap="gray" if len(img_np.shape) == 2 else None)
            if show_saliency:
                saliency_norm = (saliency - saliency.min()) / (
                    saliency.max() - saliency.min() + 1e-8
                )
                ax.imshow(saliency_norm, cmap="jet", alpha=0.5)
                title_suffix = f"Saliency: {saliency.sum():.2e}"
            ax.set_title(f"{key}\n({title_suffix})")
            ax.axis("off")

    # Plot state saliency if available
    if has_state:
        state_grad = state_saliency[0].numpy()

        # Bar chart
        ax_state = fig.add_subplot(gs[n_images, :2])
        bars = ax_state.bar(range(len(state_grad)), state_grad)
        ax_state.set_xlabel("State Dimension")
        ax_state.set_ylabel("Absolute Gradient")
        ax_state.set_title(
            f"observation.state Saliency (Total: {total_state_saliency:.2e})"
        )
        ax_state.grid(True, alpha=0.3)
        max_grad = state_grad.max()
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(state_grad[i] / (max_grad + 1e-8)))

        # Pie chart
        ax_compare = fig.add_subplot(gs[n_images, 2])
        if total_image_saliency + total_state_saliency > 0:
            ax_compare.pie(
                [total_image_saliency, total_state_saliency],
                labels=["Images", "State"],
                colors=["#ff9999", "#66b3ff"],
                autopct="%1.1f%%",
                startangle=90,
            )
            ax_compare.set_title("Total Saliency Comparison")
        else:
            ax_compare.text(0.5, 0.5, "No gradients", ha="center", va="center")
            ax_compare.axis("off")

        # Top-N comparison bar chart
        ax_topn = fig.add_subplot(gs[n_images + 1, :])
        n_top = 10  # Number of top entries to show

        # Get top n state values
        state_flat = state_saliency[0].flatten()
        top_state_values, top_state_indices = state_flat.topk(
            min(n_top, len(state_flat))
        )
        top_state_values = top_state_values.numpy()

        # Get top n image pixel values
        all_image_pixels = np.concatenate(
            [sal[0].flatten().numpy() for sal in image_saliency_dict.values()]
        )
        top_img_indices = np.argpartition(all_image_pixels, -n_top)[-n_top:]
        top_img_indices = top_img_indices[
            np.argsort(all_image_pixels[top_img_indices])[::-1]
        ]
        top_img_values = all_image_pixels[top_img_indices]

        # Create side-by-side bars
        x_positions = np.arange(n_top)
        width = 0.4
        ax_topn.bar(
            x_positions - width / 2,
            top_img_values,
            width,
            label="Image Pixels",
            color="#ff9999",
            alpha=0.8,
        )
        ax_topn.bar(
            x_positions + width / 2,
            top_state_values,
            width,
            label="State Dims",
            color="#66b3ff",
            alpha=0.8,
        )

        ax_topn.set_xlabel("Rank")
        ax_topn.set_ylabel("Saliency Value")
        ax_topn.set_title(f"Top {n_top} Most Salient Inputs Comparison")
        ax_topn.set_xticks(x_positions)
        ax_topn.set_xticklabels([f"#{i + 1}" for i in range(n_top)])
        ax_topn.legend()
        ax_topn.grid(True, alpha=0.3, axis="y")

    # Title and save
    title = f"Saliency Analysis - Episode {episode_idx}, Frame {frame_idx}"
    if has_state:
        ratio = total_state_saliency / (total_image_saliency + 1e-8)
        title += f"\nState/Image Saliency Ratio: {ratio:.2f}"
    fig.suptitle(title, fontsize=14, y=0.995)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Log statistics
    logging.info(f"Total image saliency: {total_image_saliency:.2e}")
    logging.info(f"Total state saliency: {total_state_saliency:.2e}")
    if total_image_saliency > 0:
        ratio = total_state_saliency / total_image_saliency
        logging.info(f"State/Image ratio: {ratio:.2f}")
        if ratio > 10:
            logging.warning(
                "Policy heavily relies on state - images may be underutilized"
            )
        elif ratio < 0.1:
            logging.warning(
                "Policy heavily relies on images - state may be underutilized"
            )
