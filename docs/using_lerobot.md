# Using Lerobot Directly

Our framework introduces some additional patches and default arguments for `lerobot`. Since our dataset conversion creates training data in the lerobot format, you could also use lerobot directly, i.e.

```bash
lerobot-train --policy.path=lerobot/diffusion --dataset.root=<DATAPATH> --dataset.repo_id=local --policy.push_to_hub=false --batch_size=96--steps=200000 < any additional settings >
```

Some Pitfalls are the need for a `dataset.repo_id` to correctly load local data.

If you want to enable wandb tracking, you also ned ot specify `--wandb.enable=True`.