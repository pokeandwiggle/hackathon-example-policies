import torch
import numpy as np
from unittest.mock import Mock
from scipy.spatial.transform import Rotation as R

# Assuming these are imported from your actual modules
from example_policies.robot_deploy.action_translator import ActionTranslator, ActionMode
from example_policies.data_ops.utils.geometric import quat_mul_torch


class TestDeltaTCPRotationIntegration:
    """Test suite specifically for delta TCP rotation integration."""

    def setup_method(self):
        """Setup for each test method."""
        # Mock configuration for delta TCP mode
        self.cfg = Mock()
        self.cfg.output_features = {"action": Mock()}
        self.cfg.output_features["action"].shape = [14]  # Delta TCP shape
        self.cfg.metadata = {
            "features": {
                "action": {
                    "names": [
                        "delta_tcp_left_pos_x",
                        "delta_tcp_left_pos_y",
                        "delta_tcp_left_pos_z",
                        "delta_tcp_left_rot_x",
                        "delta_tcp_left_rot_y",
                        "delta_tcp_left_rot_z",
                        "delta_tcp_right_pos_x",
                        "delta_tcp_right_pos_y",
                        "delta_tcp_right_pos_z",
                        "delta_tcp_right_rot_x",
                        "delta_tcp_right_rot_y",
                        "delta_tcp_right_rot_z",
                        "left_gripper",
                        "right_gripper",
                    ]
                },
                "observation.state": {
                    "names": [
                        "tcp_left_pos_x",
                        "tcp_left_pos_y",
                        "tcp_left_pos_z",
                        "tcp_left_quat_x",
                        "tcp_left_quat_y",
                        "tcp_left_quat_z",
                        "tcp_left_quat_w",
                        "tcp_right_pos_x",
                        "tcp_right_pos_y",
                        "tcp_right_pos_z",
                        "tcp_right_quat_x",
                        "tcp_right_quat_y",
                        "tcp_right_quat_z",
                        "tcp_right_quat_w",
                    ]
                },
            }
        }

        self.translator = ActionTranslator(self.cfg)
        assert self.translator.action_mode == ActionMode.DELTA_TCP

    def create_observation_with_pose(self, left_pos, left_quat, right_pos, right_quat):
        """Helper to create observation dict with specific poses."""
        state = torch.cat(
            [
                torch.tensor(left_pos, dtype=torch.float32),
                torch.tensor(left_quat, dtype=torch.float32),
                torch.tensor(right_pos, dtype=torch.float32),
                torch.tensor(right_quat, dtype=torch.float32),
            ]
        ).unsqueeze(
            0
        )  # Add batch dimension

        return {"observation.state": state}

    def create_delta_action(
        self,
        left_pos_delta=None,
        left_rot_delta=None,
        right_pos_delta=None,
        right_rot_delta=None,
        grippers=None,
    ):
        """Helper to create delta TCP action tensor."""
        # Default to zeros
        left_pos_delta = left_pos_delta or [0.0, 0.0, 0.0]
        left_rot_delta = left_rot_delta or [0.0, 0.0, 0.0]
        right_pos_delta = right_pos_delta or [0.0, 0.0, 0.0]
        right_rot_delta = right_rot_delta or [0.0, 0.0, 0.0]
        grippers = grippers or [0.0, 0.0]  # Note: these get inverted in translate()

        action = torch.tensor(
            [
                left_pos_delta
                + left_rot_delta
                + right_pos_delta
                + right_rot_delta
                + grippers
            ],
            dtype=torch.float32,
        )

        return action

    def test_zero_delta_rotations_preserve_pose(self):
        """Test that zero delta rotations preserve the current pose."""
        # Initial pose: identity quaternions
        initial_obs = self.create_observation_with_pose(
            left_pos=[1.0, 0.0, 0.5],
            left_quat=[0.0, 0.0, 0.0, 1.0],
            right_pos=[-1.0, 0.0, 0.5],
            right_quat=[0.0, 0.0, 0.0, 1.0],
        )

        # Zero delta action
        zero_action = self.create_delta_action()

        # First call establishes initial state
        result1 = self.translator.translate(zero_action, initial_obs)

        # Second call with zero deltas should preserve pose
        result2 = self.translator.translate(zero_action, initial_obs)

        # Extract quaternions (indices 3-6 for left, 10-13 for right in 16-dim output)
        left_quat_1 = result1[0, 3:7]
        right_quat_1 = result1[0, 10:14]
        left_quat_2 = result2[0, 3:7]
        right_quat_2 = result2[0, 10:14]

        torch.testing.assert_close(left_quat_1, left_quat_2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(right_quat_1, right_quat_2, atol=1e-6, rtol=1e-6)

    def test_very_small_delta_rotations(self):
        """Test very small delta rotations (the main bug case)."""
        initial_obs = self.create_observation_with_pose(
            left_pos=[0.0, 0.0, 0.0],
            left_quat=[0.0, 0.0, 0.0, 1.0],
            right_pos=[0.0, 0.0, 0.0],
            right_quat=[0.0, 0.0, 0.0, 1.0],
        )

        # Very small rotations that should barely change the pose
        small_deltas = [
            [1e-6, 0.0, 0.0],  # Tiny X rotation
            [0.0, 1e-6, 0.0],  # Tiny Y rotation
            [0.0, 0.0, 1e-6],  # Tiny Z rotation
            [1e-5, 1e-5, 1e-5],  # Tiny combined rotation
            [1e-4, -2e-4, 3e-4],  # Small but above numerical noise
        ]

        for i, left_rot_delta in enumerate(small_deltas):
            action = self.create_delta_action(left_rot_delta=left_rot_delta)

            # Reset translator state
            self.translator.last_action = None

            result = self.translator.translate(action, initial_obs)
            left_quat_result = result[0, 3:7]

            # For very small rotations, quaternion should be close to identity
            identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

            # Allow some tolerance but should be very close to identity
            torch.testing.assert_close(
                left_quat_result,
                identity_quat,
                atol=1e-3,
                rtol=1e-3,
                msg=f"Failed for case {i} with delta {left_rot_delta}",
            )

            # Quaternion should be normalized
            norm = torch.norm(left_quat_result)
            torch.testing.assert_close(
                norm,
                torch.tensor(1.0),
                atol=1e-6,
                rtol=1e-6,
                msg=f"Norm failed for case {i} with delta {left_rot_delta}",
            )

    def test_accumulation_of_small_rotations(self):
        """Test that small rotations accumulate correctly over multiple steps."""
        initial_obs = self.create_observation_with_pose(
            left_pos=[0.0, 0.0, 0.0],
            left_quat=[0.0, 0.0, 0.0, 1.0],
            right_pos=[0.0, 0.0, 0.0],
            right_quat=[0.0, 0.0, 0.0, 1.0],
        )

        # Apply many small rotations that should add up to a significant rotation
        small_delta = [1e-3, 0.0, 0.0]  # Small but not tiny
        num_steps = 100
        expected_total_rotation = (
            num_steps * 1e-3
        )  # Should be 0.1 radians ≈ 5.7 degrees

        for step in range(num_steps):
            action = self.create_delta_action(left_rot_delta=small_delta)
            result = self.translator.translate(action, initial_obs)

        # Final quaternion should represent the accumulated rotation
        final_left_quat = result[0, 3:7]

        # Convert back to axis-angle to verify accumulated rotation
        # Using scipy for ground truth
        final_rotation = R.from_quat(final_left_quat.numpy())  # scipy uses xyzw
        final_rotvec = final_rotation.as_rotvec()

        # Should be close to expected total rotation around X-axis
        expected_rotvec = np.array([expected_total_rotation, 0.0, 0.0])
        np.testing.assert_allclose(final_rotvec, expected_rotvec, atol=1e-2, rtol=1e-2)

    def test_mixed_small_and_large_rotations(self):
        """Test mixture of small and large delta rotations."""
        initial_obs = self.create_observation_with_pose(
            left_pos=[0.0, 0.0, 0.0],
            left_quat=[0.0, 0.0, 0.0, 1.0],
            right_pos=[0.0, 0.0, 0.0],
            right_quat=[0.0, 0.0, 0.0, 1.0],
        )

        rotation_sequence = [
            [0.1, 0.0, 0.0],  # Normal rotation
            [1e-5, 0.0, 0.0],  # Very small rotation
            [0.0, 0.05, 0.0],  # Another normal rotation
            [1e-6, 1e-6, 1e-6],  # Tiny rotation
            [0.0, 0.0, -0.08],  # Normal rotation in opposite direction
        ]

        accumulated_rotation = np.array([0.0, 0.0, 0.0])

        for i, rot_delta in enumerate(rotation_sequence):
            action = self.create_delta_action(left_rot_delta=rot_delta)
            result = self.translator.translate(action, initial_obs)

            # Track expected accumulated rotation
            accumulated_rotation += np.array(rot_delta)

            # Verify quaternion is normalized
            left_quat = result[0, 3:7]
            norm = torch.norm(left_quat)
            torch.testing.assert_close(norm, torch.tensor(1.0), atol=1e-6, rtol=1e-6)

        # Final quaternion should approximately match accumulated rotation
        final_left_quat = result[0, 3:7]
        final_rotation = R.from_quat(final_left_quat.numpy())
        final_rotvec = final_rotation.as_rotvec()

        # For small total rotations, this should be quite accurate
        np.testing.assert_allclose(
            final_rotvec, accumulated_rotation, atol=1e-2, rtol=1e-2
        )

    def test_bilateral_small_rotations(self):
        """Test small rotations on both left and right arms simultaneously."""
        initial_obs = self.create_observation_with_pose(
            left_pos=[0.5, 0.0, 0.0],
            left_quat=[0.0, 0.0, 0.0, 1.0],
            right_pos=[-0.5, 0.0, 0.0],
            right_quat=[0.0, 0.0, 0.0, 1.0],
        )

        # Small rotations on both arms
        action = self.create_delta_action(
            left_rot_delta=[1e-4, 2e-4, -1e-4], right_rot_delta=[-2e-4, 1e-4, 3e-4]
        )

        result = self.translator.translate(action, initial_obs)

        # Both quaternions should be close to identity but slightly rotated
        left_quat = result[0, 3:7]
        right_quat = result[0, 10:14]

        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

        # Should be close to identity but not exactly
        assert not torch.allclose(left_quat, identity_quat, atol=1e-6)
        assert not torch.allclose(right_quat, identity_quat, atol=1e-6)

        # But still very close
        torch.testing.assert_close(left_quat, identity_quat, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(right_quat, identity_quat, atol=1e-2, rtol=1e-2)

        # Both should be normalized
        left_norm = torch.norm(left_quat)
        right_norm = torch.norm(right_quat)
        torch.testing.assert_close(left_norm, torch.tensor(1.0), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(right_norm, torch.tensor(1.0), atol=1e-6, rtol=1e-6)

    def test_numerical_stability_edge_cases(self):
        """Test edge cases that could cause numerical instability."""
        initial_obs = self.create_observation_with_pose(
            left_pos=[0.0, 0.0, 0.0],
            left_quat=[0.0, 0.0, 0.0, 1.0],
            right_pos=[0.0, 0.0, 0.0],
            right_quat=[0.0, 0.0, 0.0, 1.0],
        )

        edge_cases = [
            [0.0, 0.0, 0.0],  # Exactly zero
            [1e-8, 0.0, 0.0],  # Extremely tiny
            [1e-7, 1e-7, 1e-7],  # Tiny isotropic
            [1e-15, 0.0, 0.0],  # Near machine epsilon
            [-1e-5, 1e-5, -1e-5],  # Small alternating signs
        ]

        for i, rot_delta in enumerate(edge_cases):
            action = self.create_delta_action(left_rot_delta=rot_delta)

            # Reset state
            self.translator.last_action = None

            # Should not crash or produce invalid results
            try:
                result = self.translator.translate(action, initial_obs)
                left_quat = result[0, 3:7]

                # Quaternion should be valid and normalized
                norm = torch.norm(left_quat)
                torch.testing.assert_close(
                    norm,
                    torch.tensor(1.0),
                    atol=1e-6,
                    rtol=1e-6,
                    msg=f"Norm failed for case {i} with delta {rot_delta}",
                )

                # Should not contain NaN or inf
                assert torch.isfinite(
                    left_quat
                ).all(), f"Invalid values for case {i} with delta {rot_delta}"

            except Exception as e:
                pytest.fail(f"Edge case {i} with delta {rot_delta} failed: {e}")

    def test_consistency_with_reference_implementation(self):
        """Test consistency with a reference implementation using scipy."""
        initial_obs = self.create_observation_with_pose(
            left_pos=[0.0, 0.0, 0.0],
            left_quat=[0.1, 0.2, 0.3, 0.9],  # Some initial rotation
            right_pos=[0.0, 0.0, 0.0],
            right_quat=[0.0, 0.0, 0.0, 1.0],
        )

        # Normalize the initial quaternion
        initial_left_quat = np.array([0.1, 0.2, 0.3, 0.9])
        initial_left_quat = initial_left_quat / np.linalg.norm(initial_left_quat)
        initial_obs["observation.state"][0, 3:7] = torch.tensor(
            initial_left_quat, dtype=torch.float32
        )

        test_deltas = [
            [1e-4, 0.0, 0.0],
            [0.0, 2e-4, 0.0],
            [1e-3, -1e-3, 5e-4],
        ]

        for delta in test_deltas:
            action = self.create_delta_action(left_rot_delta=delta)

            # Reset state
            self.translator.last_action = None

            # Our implementation
            result = self.translator.translate(action, initial_obs)
            our_quat = result[0, 3:7].numpy()

            # Reference implementation using scipy
            initial_rotation = R.from_quat(initial_left_quat)  # scipy uses xyzw
            delta_rotation = R.from_rotvec(delta)
            expected_rotation = initial_rotation * delta_rotation
            expected_quat = expected_rotation.as_quat()  # xyzw format

            # Should match within reasonable tolerance
            np.testing.assert_allclose(
                our_quat,
                expected_quat,
                atol=1e-4,
                rtol=1e-4,
                err_msg=f"Failed for delta {delta}",
            )


# Example usage and additional debugging utilities
if __name__ == "__main__":
    print("Running Delta TCP Rotation Integration Tests...")

    test_suite = TestDeltaTCPRotationIntegration()

    # Run key tests
    test_suite.setup_method()

    try:
        test_suite.test_zero_delta_rotations_preserve_pose()
        print("✓ Zero delta rotations test passed")
    except Exception as e:
        print(f"✗ Zero delta rotations test failed: {e}")

    try:
        test_suite.setup_method()  # Reset
        test_suite.test_very_small_delta_rotations()
        print("✓ Very small delta rotations test passed")
    except Exception as e:
        print(f"✗ Very small delta rotations test FAILED: {e}")
        print("This is likely the bug you're trying to fix!")

    try:
        test_suite.setup_method()  # Reset
        test_suite.test_numerical_stability_edge_cases()
        print("✓ Numerical stability edge cases test passed")
    except Exception as e:
        print(f"✗ Numerical stability test failed: {e}")

    print("\nTo run full test suite: pytest -v this_file.py")
