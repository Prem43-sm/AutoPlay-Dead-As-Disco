import unittest

from app.config import CaptureConfig, CaptureRegion


class CaptureConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        CaptureConfig().validate()

    def test_region_requires_positive_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            CaptureRegion(0, 0, 0, 720).validate()

    def test_region_must_not_start_before_monitor_origin(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            CaptureRegion(-1, 0, 1280, 720).validate()

    def test_monitor_index_must_be_one_based(self) -> None:
        with self.assertRaisesRegex(ValueError, "1 or greater"):
            CaptureConfig(monitor_index=0).validate()

    def test_fps_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            CaptureConfig(max_fps=0).validate()

    def test_fps_must_be_finite(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            CaptureConfig(max_fps=float("nan")).validate()


if __name__ == "__main__":
    unittest.main()
