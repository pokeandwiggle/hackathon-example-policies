from example_policies.data_ops.merge_lerobot import merge_datasets

if __name__ == "__main__":
    import pathlib

    path_step_1_and_2 = pathlib.Path("data/lerobot/step_1_tcp")
    path_step_3 = pathlib.Path("data/lerobot/step_1_tcp_recovery")
    output_path = pathlib.Path("data/lerobot/step_1_tcp_with_recovery")
    merge_datasets(
        [path_step_1_and_2, path_step_3],
        output_path,
    )
