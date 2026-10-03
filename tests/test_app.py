import unittest
from unittest.mock import Mock, patch

from PIL import Image

import app


class TestMemeGenerator(unittest.TestCase):
    def test_cpu_device_config(self):
        with patch("app.torch.cuda.is_available", return_value=False):
            self.assertEqual(app.get_device_config(), ("cpu", app.torch.float32))

    def test_cuda_device_config(self):
        with patch("app.torch.cuda.is_available", return_value=True):
            self.assertEqual(app.get_device_config(), ("cuda", app.torch.float16))

    def test_add_text_to_image_changes_pixels(self):
        image = Image.new("RGB", (256, 256), "gray")
        before = image.copy()

        app.add_text_to_image(image, "test")

        self.assertNotEqual(list(image.getdata()), list(before.getdata()))

    def test_generate_images_uses_seed(self):
        pipeline = Mock()
        pipeline.return_value.images = ["image"]

        result = app.generate_images("prompt", pipeline, 1, seed=42)

        self.assertEqual(result, ["image"])
        pipeline.assert_called_once()
        _, kwargs = pipeline.call_args
        self.assertIsNotNone(kwargs["generator"])

    def test_generate_images_without_seed(self):
        pipeline = Mock()
        pipeline.return_value.images = ["image"]

        result = app.generate_images("prompt", pipeline, 1)

        self.assertEqual(result, ["image"])
        _, kwargs = pipeline.call_args
        self.assertIsNone(kwargs["generator"])


if __name__ == "__main__":
    unittest.main()
