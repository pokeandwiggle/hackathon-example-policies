from example_policies.data_ops.merge_lerobot import merge_datasets

if __name__ == "__main__":
    import pathlib

    path_step_1 = pathlib.Path("data/lerobot/step_1")
    path_step_2 = pathlib.Path("data/lerobot/step_2")
    output_path = pathlib.Path("data/lerobot/step_1_and_2")
    merge_datasets(
        [path_step_1, path_step_2],
        output_path,
    )
