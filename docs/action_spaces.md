# Action Space Configuration ⚙️

The action space for the policy is determined during the dataset conversion process. The choice of the `--action-level` flag specifies which ROS topics are used as the action source and defines the shape and content of the `batch["action"]` tensor available during training.

The following table details the available action level configurations:

| Action Source                   | ROS Topics                                                        | Frequency       | Dataset Conversion Flag      | `batch["action"]` Format                                                                                                             |
| :------------------------------ | :---------------------------------------------------------------- | :-------------- | :--------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| **VR Controller Signal**        | `/desired_pose_left`<br>`/desired_pose_right`                     | ~2Hz (Very Low) | `--action-level=teleop`      | **Shape**: `(16,)`<br>**Content**: `left_xyz`, `left_xyzw`, `right_xyz`, `right_xyzw`, `left_gripper`, `right_gripper`               |
| **Cartesian Waypoints**         | `/desired_pose_twist_left`<br>`/desired_pose_twist_right`         | ~1000Hz (High)  | `--action-level=tcp`         | **Shape**: `(16,)`<br>**Content**: `left_xyz`, `left_xyzw`, `right_xyz`, `right_xyzw`, `left_gripper`, `right_gripper`               |
| **Cartesian Waypoints (delta)** | `/desired_pose_twist_left`<br>`/desired_pose_twist_right`         | ~1000Hz (High)  | `--action-level=delta_tcp`   | **Shape**: `(14,)`<br>**Content**: `left_d_xyz`, `left_d_rot_xyz`, `right_d_xyz`, `right_d_rot_xyz`, `left_gripper`, `right_gripper` |
| **Joint Waypoints**             | `/left_desired_joint_waypoint`<br>`/right_desired_joint_waypoint` | ~500Hz (High)   | `--action-level=joint`       | **Shape**: `(16,)`<br>**Content**: `left_joint_pos[7]`, `right_joint_pos[7]`, `left_gripper`, `right_gripper`                        |
| **Joint Waypoints (delta)**     | `/left_desired_joint_waypoint`<br>`/right_desired_joint_waypoint` | ~500Hz (High)   | `--action-level=delta_joint` | **Shape**: `(16,)`<br>**Content**: `left_joint_d_pos[7]`, `right_joint_d_pos[7]`, `left_gripper`, `right_gripper`                    |

---

### Action Level Descriptions

*   **`teleop`**: Uses low-frequency signals directly from a VR controller. The action represents the absolute target end-effector pose.
*   **`tcp`**: Uses high-frequency Cartesian waypoints. The action represents the absolute target end-effector pose, identical in format to `teleop`.
*   **`delta_tcp`**: Uses high-frequency Cartesian waypoints but represents the action as a delta from the current pose. The rotation is represented by a 3D axis-angle vector instead of a quaternion.
*   **`joint`**: Uses high-frequency joint waypoints. The action represents the absolute target position for each of the 7 joints per arm.
*   **`delta_joint`**: Uses high-frequency joint waypoints but represents the action as a delta from the current joint positions.

---

### 🧠 Key Concepts for Training

#### Cartesian Control (`tcp` / `delta_tcp`)

When working with Cartesian end-effector control, it's crucial to handle rotations correctly. A simple Mean-Squared-Error (MSE) loss on quaternion components is often ineffective because the numerical difference between quaternions does not correspond to the actual rotational distance.

💡 To address this, we provide SO(3)-aware implementations that correctly compute the shortest rotational distance (geodesic) between two quaternions:
*   `so3_integrated_act`
*   `so3_integrated_diffusion`

For a deeper dive into this topic, we recommend researching "Riemannian geometry for robotics."

#### Joint Level Control (`joint` / `delta_joint`)

Learning joint-level control is generally more straightforward and does not require special model implementations for the loss function.
> **Note**: This feature is still under construction.

### 📝 Dataset Conversion Notes

*   **Matching States**: Ensure you include the same state information during conversion that your policy will need for training (e.g., include `--include-joints` for joint-level control).
*   **Delta Control**: For delta-based action levels, the model typically requires access to the last executed command to compute the next delta.
*   **Default Configuration**: The default `PipelineConfig` is configured for `delta_tcp`.

### 🚀 Policy Deployment

When using our framework, the deployment script automatically detects the `action_level` the policy was trained on and correctly translates the actions for the real robot hardware.

### 🔗 See Also

*   [PipelineConfig](../src/example_policies/data_ops/config/pipeline_config.py)
*   [RosTopics](../src/example_policies/data_ops/config/rosbag_topics.py)
