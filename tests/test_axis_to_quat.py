import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from example_policies.data_ops.utils.geometric import axis_angle_to_quat_torch

# Assuming axis_angle_to_quat_torch function is defined above...


class TestAxisAngleToQuatTorch:
    """
    A test suite focused on the core mathematical correctness of the conversion.
    """

    def test_zero_rotation(self):
        """Test that zero rotation gives identity quaternion."""
        aa = torch.zeros(3)
        quat = axis_angle_to_quat_torch(aa)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
        torch.testing.assert_close(quat, expected)

    def test_very_small_rotations(self):
        """Test very small rotations use Taylor approximation correctly."""
        aa = torch.tensor([1e-5, -2e-5, 5e-5])
        quat = axis_angle_to_quat_torch(aa, eps=1e-3)
        # For small angles, expect quat ≈ [aa/2, 1]
        expected = torch.tensor([0.5e-5, -1e-5, 2.5e-5, 1.0])
        # The output quat won't be perfectly normalized here, so we check the components
        torch.testing.assert_close(quat, expected, atol=1e-9, rtol=1e-9)

    def test_different_dtypes(self):
        """Test with different tensor dtypes."""
        aa = torch.tensor([0.1, 0.2, 0.3])

        for dtype in [torch.float32, torch.float64]:
            aa_typed = aa.to(dtype)
            quat = axis_angle_to_quat_torch(aa_typed)
            assert quat.dtype == dtype

            # Should still be unit quaternion
            norm = torch.norm(quat)
            torch.testing.assert_close(
                norm, torch.tensor(1.0, dtype=dtype), atol=1e-5, rtol=1e-5
            )

    def test_canonical_rotations(self):
        """Test 180-degree rotation around main axes."""
        # 180° rotation around X-axis
        aa_x = torch.tensor(
            [torch.pi, 0, 0],
        )
        quat_x = axis_angle_to_quat_torch(aa_x)
        expected_x = torch.tensor([1.0, 0.0, 0.0, 0.0])
        torch.testing.assert_close(quat_x, expected_x, atol=1e-6, rtol=1e-6)

        # 90° rotation around Y-axis
        aa_y = torch.tensor([0, torch.pi / 2, 0])
        quat_y = axis_angle_to_quat_torch(aa_y)
        expected_y = torch.tensor(
            [0.0, np.sin(torch.pi / 4), 0.0, np.cos(torch.pi / 4)], dtype=torch.float32
        )
        torch.testing.assert_close(quat_y, expected_y, atol=1e-6, rtol=1e-6)

    def test_arbitrary_rotations_with_scipy(self):
        """Test arbitrary rotations by comparing with a trusted library."""
        aa_list = [0.1, -0.5, 1.2]
        aa = torch.tensor(aa_list, dtype=torch.float32)
        quat = axis_angle_to_quat_torch(aa)

        # Compare with scipy (which returns xyzw)
        scipy_quat = R.from_rotvec(aa_list).as_quat()
        expected = torch.tensor(scipy_quat, dtype=torch.float32)
        torch.testing.assert_close(quat, expected)

    def test_quaternion_norm(self):
        """Test that output quaternions have unit norm (for non-Taylor cases)."""
        aa = torch.tensor([0.2, 0.3, 0.4])
        quat = axis_angle_to_quat_torch(aa)
        norm = torch.norm(quat)
        torch.testing.assert_close(norm, torch.tensor(1.0))

    def test_eps_boundary(self):
        """Test behavior around the small-angle approximation boundary."""
        eps = 1e-3

        # Just below eps -> uses Taylor approx. q ≈ [aa/2, 1]
        aa_small = torch.tensor([eps * 0.99, 0.0, 0.0])
        quat_small = axis_angle_to_quat_torch(aa_small, eps=eps)
        # Note: The Taylor approximation is not perfectly unit-norm
        assert torch.allclose(
            quat_small, torch.tensor([eps * 0.99 / 2, 0.0, 0.0, 1.0]), atol=1e-9
        )

        # Just above eps -> uses standard trig conversion
        aa_large = torch.tensor([eps * 1.01, 0.0, 0.0])
        quat_large = axis_angle_to_quat_torch(aa_large, eps=eps)
        angle = torch.norm(aa_large)
        expected_trig = torch.tensor(
            [torch.sin(angle / 2), 0.0, 0.0, torch.cos(angle / 2)]
        )
        torch.testing.assert_close(quat_large, expected_trig)


# Example usage and additional validation
if __name__ == "__main__":
    # Run a few quick tests
    test_suite = TestAxisAngleToQuatTorch()

    print("Running axis-angle to quaternion conversion tests...")

    test_suite.test_zero_rotation()
    print("✓ Zero rotation test passed")

    test_suite.test_very_small_rotations()
    print("✓ Very small rotations test passed")

    test_suite.test_different_dtypes()
    print("✓ Different dtypes test passed")

    test_suite.test_canonical_rotations()
    print("✓ Canonical rotations test passed")

    test_suite.test_eps_boundary()
    print("✓ Eps boundary test passed")

    print("\nAll tests passed! 🎉")

    # To run with pytest: pytest -v this_file.py
