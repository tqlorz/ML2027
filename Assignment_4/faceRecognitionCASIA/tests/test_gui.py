import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from config import config
from face_verification import VerificationResult
from gui import FaceVerificationWindow, configure_font


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)
        configure_font(cls.app)
        cls.face = Path(config.train_root) / "0000045/001.jpg"

    def setUp(self):
        self.window = FaceVerificationWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.wait_until(lambda: not self.window.busy)
        self.window.close()
        self.app.processEvents()

    def wait_until(self, condition, timeout=20):
        if condition():
            return
        loop = QEventLoop()
        poll = QTimer()
        poll.timeout.connect(lambda: loop.quit() if condition() else None)
        poll.start(10)
        deadline = QTimer()
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        deadline.start(timeout * 1000)
        loop.exec()
        poll.stop()
        deadline.stop()
        self.assertTrue(condition(), "GUI operation timed out")

    def select_pair(self, second=None):
        self.assertTrue(self.window.load_image(0, self.face))
        self.assertTrue(self.window.load_image(1, second or self.face))

    def test_selection_invalid_file_and_clearing(self):
        self.assertFalse(self.window.verify_button.isEnabled())
        self.select_pair()
        self.assertTrue(self.window.verify_button.isEnabled())
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "corrupt.jpg"
            invalid.write_text("not an image")
            self.assertFalse(self.window.load_image(0, invalid))
        self.assertEqual(self.window.cards[0].path, self.face.resolve())
        self.assertEqual(self.window.result_title.text(), "无法加载图片")
        self.window.clear()
        self.assertFalse(self.window.verify_button.isEnabled())
        self.assertTrue(all(card.path is None for card in self.window.cards))

    def test_real_backend_same_different_and_no_face(self):
        other = Path(config.train_root) / "0000099/001.jpg"
        for second, expected in [(self.face, "是同一个人"), (other, "不是同一个人")]:
            self.select_pair(second)
            self.window.verify_button.click()
            self.wait_until(lambda: not self.window.busy)
            self.assertEqual(self.window.result_title.text(), expected)
            self.assertIn("余弦相似度", self.window.result_scores.text())
        with tempfile.TemporaryDirectory() as directory:
            blank = Path(directory) / "blank.png"
            Image.new("RGB", (128, 128), "white").save(blank)
            self.select_pair(blank)
            self.window.start_verification()
            self.wait_until(lambda: not self.window.busy)
            self.assertEqual(self.window.result_title.text(), "未检测到清晰人脸")

    def test_worker_keeps_ui_responsive_and_recovers_from_error(self):
        release = threading.Event()

        def delayed_error(*paths):
            release.wait(2)
            raise ValueError("Threshold metadata mismatch for checkpoint_sha256")

        self.window.verifier = delayed_error
        self.select_pair()
        ticks = []
        timer = QTimer(self.window)
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(10)
        self.window.start_verification()
        try:
            self.assertFalse(self.window.verify_button.isEnabled())
            self.wait_until(lambda: len(ticks) >= 2)
            self.assertTrue(ticks, "UI event loop stalled during worker execution")
            self.assertFalse(self.window.load_image(0, self.face))
        finally:
            release.set()
        self.wait_until(lambda: not self.window.busy)
        timer.stop()
        self.assertEqual(self.window.result_title.text(), "验证失败")
        self.assertIn("模型与验证阈值不匹配", self.window.result_detail.text())
        self.assertTrue(self.window.verify_button.isEnabled())
        self.window.verifier = lambda *paths: VerificationResult(1, 0.9, 0.4, True)
        self.window.start_verification()
        self.wait_until(lambda: not self.window.busy)
        self.assertEqual(self.window.result_title.text(), "是同一个人")
        self.window.load_image(1, self.face)
        self.assertEqual(self.window.result_title.text(), "等待验证")

    def test_closing_during_inference_waits_for_worker(self):
        release = threading.Event()

        def delayed_result(*paths):
            release.wait(2)
            return VerificationResult(1, 0.9, 0.4, True)

        self.window.verifier = delayed_result
        self.select_pair()
        self.window.start_verification()
        self.window.close()
        self.assertTrue(self.window.closing)
        self.assertFalse(self.window.isVisible())
        release.set()
        self.wait_until(lambda: not self.window.busy)
        self.assertIsNone(self.window.thread)


if __name__ == "__main__":
    unittest.main()
