# Using Lerobot Directly

Our framework introduces additional patches and default arguments for [`lerobot`](https://github.com/huggingface/lerobot). Since our dataset conversion creates training data in the `lerobot` format, you can also use `lerobot` directly for training.

## Example Usage

Here is an example command to start training with `lerobot`:

```bash
lerobot-train \
    --policy.path=lerobot/diffusion \
    --dataset.root=<DATAPATH> \
    --dataset.repo_id=local \
    --policy.push_to_hub=false \
    --batch_size=96 \
    --steps=200000 \
    # <any additional settings>
```

## Important Configuration Notes

When using `lerobot` directly, keep the following points in mind:

*   **Local Datasets**: To load a local dataset, you must specify `--dataset.repo_id=local`.
*   **W&B Logging**: To enable logging with Weights & Biases, you need to explicitly add the `--wandb.enable=True` flag.