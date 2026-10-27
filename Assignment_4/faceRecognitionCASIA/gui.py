"""Chinese desktop GUI for the FaceVerification backend; launch with python gui.py."""

import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QProgressBar, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)


COLORS = {
    "background": "#f3f6f8", "surface": "#ffffff", "ink": "#152b3a",
    "muted": "#657985", "line": "#dce5e9", "accent": "#147b73",
}


def configure_font(app):
    candidates = ("Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "PingFang SC")
    families = set(QFontDatabase.families())
    # WSLg can reuse the Windows font without installing or copying font files.
    windows_font = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    if not any(name in families for name in candidates) and windows_font.is_file():
        QFontDatabase.addApplicationFont(str(windows_font))
        families = set(QFontDatabase.families())
    family = next((name for name in candidates if name in families), app.font().family())
    app.setFont(QFont(family, 10))


def error_message(error):
    if isinstance(error, UnidentifiedImageError):
        return "图片无法读取，请重新选择有效的 JPG、PNG 或其他图片文件。"
    if isinstance(error, FileNotFoundError):
        return "找不到图片、模型或阈值文件。请确认图片仍可访问，并按 README 配置模型与阈值。"
    if isinstance(error, ImportError):
        return "缺少运行依赖，请按照 README 安装 requirements.txt 中的依赖。"
    if isinstance(error, ValueError) and "metadata mismatch" in str(error):
        return "模型与验证阈值不匹配，请使用 calibrate.py 为当前模型重新标定。"
    return f"验证未能完成：{error}"


class ImagePreview(QWidget):
    clicked = Signal()

    def __init__(self):
        super().__init__()
        self.pixmap = None
        self.setMinimumSize(180, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#edf2f5"))
        painter.drawRoundedRect(QRectF(self.rect()), 10, 10)
        if self.pixmap is not None:
            target = QRectF(12, 12, self.width() - 24, self.height() - 24)
            scaled = self.pixmap.scaled(target.size().toSize(), Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
            x, y = (self.width() - scaled.width()) / 2, (self.height() - scaled.height()) / 2
            painter.drawPixmap(int(x), int(y), scaled)
        else:
            cx, cy = self.width() / 2, self.height() / 2
            painter.setPen(QPen(QColor("#91a7b1"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy - 38), 21, 21)
            shoulders = QPainterPath(QPointF(cx - 43, cy + 24))
            shoulders.cubicTo(cx - 43, cy - 15, cx + 43, cy - 15, cx + 43, cy + 24)
            painter.drawPath(shoulders)
            painter.setPen(QColor(COLORS["muted"]))
            painter.drawText(QRectF(0, cy + 48, self.width(), 30), Qt.AlignmentFlag.AlignCenter,
                             "点击选择一张照片")
        painter.end()

    def mousePressEvent(self, event):
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ImageCard(QFrame):
    choose_requested = Signal()

    def __init__(self, number):
        super().__init__()
        self.path = None
        self.setObjectName("imageCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        heading = QLabel(f"图片 {number:02d}")
        heading.setObjectName("cardHeading")
        layout.addWidget(heading)
        self.preview = ImagePreview()
        self.preview.clicked.connect(self.choose_requested.emit)
        layout.addWidget(self.preview, 1)
        self.filename = QLabel("尚未选择图片")
        self.filename.setTextFormat(Qt.TextFormat.PlainText)
        self.filename.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filename.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.filename)
        self.info = QLabel("支持 JPG、PNG、BMP、WEBP 等格式")
        self.info.setObjectName("mutedSmall")
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info)
        self.choose_button = QPushButton("选择图片")
        self.choose_button.setObjectName("secondaryButton")
        self.choose_button.clicked.connect(self.choose_requested.emit)
        layout.addWidget(self.choose_button, 0, Qt.AlignmentFlag.AlignHCenter)

    def load(self, path):
        path = Path(path).expanduser().resolve()
        # Decode completely before replacing the previous valid selection.
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        dimensions = image.size
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        raw = image.tobytes("raw", "RGB")
        qimage = QImage(raw, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.path = path
        self.preview.pixmap = pixmap
        self.preview.update()
        name = path.name if len(path.name) <= 40 else path.name[:20] + "…" + path.name[-16:]
        self.filename.setText(name)
        self.filename.setToolTip(str(path))
        self.info.setText(f"{dimensions[0]} × {dimensions[1]} 像素")
        self.choose_button.setText("更换图片")

    def clear(self):
        self.path = None
        self.preview.pixmap = None
        self.preview.update()
        self.filename.setText("尚未选择图片")
        self.filename.setToolTip("")
        self.info.setText("支持 JPG、PNG、BMP、WEBP 等格式")
        self.choose_button.setText("选择图片")


class VerificationWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()

    def __init__(self, paths, checkpoint, threads, verifier=None):
        super().__init__()
        self.paths, self.checkpoint, self.threads, self.verifier = paths, checkpoint, threads, verifier

    @Slot()
    def run(self):
        try:
            if self.verifier is None:
                import torch
                from config import config
                from face_verification import compare_faces

                torch.set_num_threads(self.threads)
                if self.checkpoint is not None:
                    config.test_model = self.checkpoint
                result = compare_faces(*self.paths)
            else:
                result = self.verifier(*self.paths)
            self.succeeded.emit(result)
        except Exception as error:
            self.failed.emit(error)
        finally:
            self.finished.emit()


class FaceVerificationWindow(QMainWindow):
    def __init__(self, checkpoint=None, threads=4, verifier=None):
        super().__init__()
        self.checkpoint, self.threads, self.verifier = checkpoint, threads, verifier
        self.thread = None
        self.worker = None
        self.busy = False
        self.closing = False
        self.setWindowTitle("人脸验证 · Face Verification")
        self.resize(1060, 840)
        self.setMinimumSize(880, 760)
        self._build()
        self._styles()
        self._reset_result()

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(36, 28, 36, 24)
        layout.setSpacing(0)
        eyebrow = QLabel("FACE VERIFICATION")
        eyebrow.setObjectName("eyebrow")
        layout.addWidget(eyebrow)
        title = QLabel("人脸验证")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        description = QLabel("选择两张照片，比较是否为同一个人。")
        description.setObjectName("muted")
        layout.addWidget(description)
        layout.addSpacing(12)
        self.selection_label = QLabel("已选择 0 / 2 张图片")
        self.selection_label.setObjectName("mutedSmall")
        self.selection_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.selection_label)
        layout.addSpacing(10)
        cards = QHBoxLayout()
        cards.setSpacing(20)
        self.cards = []
        for index in range(2):
            card = ImageCard(index + 1)
            card.choose_requested.connect(lambda i=index: self.choose_image(i))
            cards.addWidget(card, 1)
            self.cards.append(card)
        layout.addLayout(cards, 1)
        layout.addSpacing(18)
        controls = QHBoxLayout()
        self.clear_button = QPushButton("清空图片")
        self.clear_button.setObjectName("secondaryButton")
        self.clear_button.clicked.connect(self.clear)
        controls.addWidget(self.clear_button)
        controls.addStretch()
        self.verify_button = QPushButton("开始验证")
        self.verify_button.setObjectName("primaryButton")
        self.verify_button.setEnabled(False)
        self.verify_button.clicked.connect(self.start_verification)
        controls.addWidget(self.verify_button)
        layout.addLayout(controls)
        layout.addSpacing(16)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        layout.addSpacing(16)
        panel = QFrame()
        panel.setObjectName("resultPanel")
        result_layout = QHBoxLayout(panel)
        result_layout.setContentsMargins(22, 18, 22, 18)
        result_layout.setSpacing(18)
        self.result_symbol = QLabel("—")
        self.result_symbol.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_symbol.setFixedSize(46, 46)
        result_layout.addWidget(self.result_symbol)
        texts = QVBoxLayout()
        texts.setSpacing(5)
        self.result_title = QLabel()
        self.result_detail = QLabel()
        self.result_detail.setTextFormat(Qt.TextFormat.PlainText)
        self.result_detail.setObjectName("muted")
        self.result_detail.setWordWrap(True)
        self.result_scores = QLabel()
        self.result_scores.setObjectName("mutedSmall")
        texts.addWidget(self.result_title)
        texts.addWidget(self.result_detail)
        texts.addWidget(self.result_scores)
        result_layout.addLayout(texts, 1)
        layout.addWidget(panel)
        layout.addSpacing(12)
        footer = QLabel("每张图片默认比较其中最大的人脸。")
        footer.setObjectName("mutedSmall")
        layout.addWidget(footer)

    def _styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f3f6f8; color: #152b3a; }
            QLabel { background: transparent; }
            QLabel#eyebrow { color: #147b73; font-size: 11px; font-weight: 700; letter-spacing: 2px; }
            QLabel#pageTitle { font-size: 32px; font-weight: 700; margin-top: 5px; margin-bottom: 5px; }
            QLabel#muted { color: #657985; font-size: 13px; }
            QLabel#mutedSmall { color: #657985; font-size: 11px; }
            QFrame#imageCard, QFrame#resultPanel { background: white; border: 1px solid #dce5e9; border-radius: 14px; }
            QLabel#cardHeading { color: #657985; font-weight: 700; font-size: 12px; margin-bottom: 2px; }
            QPushButton { border: none; border-radius: 8px; padding: 10px 22px; font-size: 13px; }
            QPushButton#primaryButton { background: #147b73; color: white; font-weight: 700; min-width: 120px; padding: 13px 28px; }
            QPushButton#primaryButton:hover { background: #10665f; }
            QPushButton#primaryButton:pressed { background: #0b554f; }
            QPushButton#primaryButton:disabled { background: #cbd7dc; color: #f7fafb; }
            QPushButton#secondaryButton { background: #eaf2f3; color: #147b73; }
            QPushButton#secondaryButton:hover { background: #deeaec; }
            QPushButton#secondaryButton:disabled { color: #9aabb3; }
            QPushButton:focus { outline: none; border: 2px solid #82bdb6; }
            QProgressBar { background: #dce5e9; border: none; border-radius: 2px; }
            QProgressBar::chunk { background: #147b73; border-radius: 2px; }
        """)

    def _set_result(self, title, detail, symbol="—", color="#152b3a", scores=""):
        self.result_title.setText(title)
        self.result_title.setStyleSheet(f"font-size: 19px; font-weight: 700; color: {color};")
        self.result_detail.setText(detail)
        self.result_symbol.setText(symbol)
        self.result_symbol.setStyleSheet(f"font-size: 28px; color: {color}; background: #edf4f4; border-radius: 23px;")
        self.result_scores.setText(scores)

    def _reset_result(self):
        self._set_result("等待验证", "选择两张图片后，点击「开始验证」。")

    def _update_selection(self):
        count = sum(card.path is not None for card in self.cards)
        self.selection_label.setText(f"已选择 {count} / 2 张图片")
        self.verify_button.setEnabled(count == 2 and not self.busy)

    def choose_image(self, index):
        if self.busy:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, f"选择第 {index + 1} 张图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.gif);;所有文件 (*)",
        )
        if path:
            self.load_image(index, path)

    def load_image(self, index, path):
        if self.busy or self.closing:
            return False
        try:
            self.cards[index].load(path)
        except (OSError, ValueError) as error:
            self._set_result("无法加载图片", error_message(error), "!", "#a74a39")
            return False
        self._reset_result()
        self._update_selection()
        return True

    def clear(self):
        if self.busy:
            return
        for card in self.cards:
            card.clear()
        self._reset_result()
        self._update_selection()

    def start_verification(self):
        if self.busy or self.closing or any(card.path is None for card in self.cards):
            return
        self.busy = True
        self._update_selection()
        self.clear_button.setEnabled(False)
        for card in self.cards:
            card.setEnabled(False)
        self.verify_button.setText("正在验证…")
        self._set_result("正在验证…", "首次验证需要加载模型，请稍候。", "…", COLORS["accent"])
        self.progress.setRange(0, 0)
        self.thread = QThread(self)
        self.worker = VerificationWorker(tuple(card.path for card in self.cards), self.checkpoint,
                                         self.threads, self.verifier)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.succeeded.connect(self._show_result)
        self.worker.failed.connect(self._show_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @Slot(object)
    def _show_result(self, result):
        if not result.faces_detected:
            self._set_result("未检测到清晰人脸", "至少一张图片未检出人脸，请换用面部清晰的照片后重试。", "!", "#9a661b")
        else:
            same = result.prediction == 1
            self._set_result(
                "是同一个人" if same else "不是同一个人",
                "两张照片中的人脸特征相近。" if same else "两张照片中的人脸特征未达到匹配阈值。",
                "✓" if same else "≠", COLORS["accent"] if same else "#9a661b",
                f"余弦相似度  {result.similarity:.3f}    /    判定阈值  {result.threshold:.3f}",
            )

    @Slot(object)
    def _show_error(self, error):
        self._set_result("验证失败", error_message(error), "!", "#a74a39")

    @Slot()
    def _thread_finished(self):
        self.busy = False
        self.thread, self.worker = None, None
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.verify_button.setText("开始验证")
        self.clear_button.setEnabled(True)
        for card in self.cards:
            card.setEnabled(True)
        self._update_selection()
        if self.closing:
            self.close()

    def closeEvent(self, event):
        if self.busy:
            # Let the active native inference finish before disposing its QThread.
            self.closing = True
            self.hide()
            event.ignore()
        else:
            event.accept()


def main():
    parser = argparse.ArgumentParser(description="双图人脸验证 GUI")
    parser.add_argument("--checkpoint", help="指定 checkpoints/ 中的权重文件名")
    parser.add_argument("--threads", type=int, default=4, help="CPU 线程数，默认 4")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads 必须为正整数")
    if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "wayland;xcb")
    app = QApplication(sys.argv[:1])
    configure_font(app)
    window = FaceVerificationWindow(checkpoint=args.checkpoint, threads=args.threads)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
