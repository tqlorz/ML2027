# 人脸验证

本项目实现 Assignment 4 要求的人脸验证接口及中文 GUI。`FaceVerification(X1, X2)` 接收两张图片的路径，检测到的人脸属于同一个人时返回 `1`，否则返回 `0`。处理流程为：使用 MTCNN 检测人脸，选择最大人脸并根据五个关键点进行对齐，通过 FaceMobileNet 提取 512 维特征向量，再将两个向量的余弦相似度与 CASIA 验证集标定的阈值比较。

## 环境配置

使用 Python 3.10 或 3.11。在本项目目录中解压 `datasets.zip`，解压后应存在 `datasets/CASIA-WebFace-Align-96` 和 `datasets/scikit_learn_data/lfw_home` 两个目录。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

如果使用 CUDA，请先安装与本机 CUDA 环境兼容的 PyTorch 2.2.2 和 torchvision 0.17.2，再安装其余依赖。

## 运行方法

以下命令均在项目目录中执行。先进入目录并激活虚拟环境：

```bash
cd Assignment_4/faceRecognitionCASIA
source .venv/bin/activate
```

### 图形界面

```bash
python gui.py
```

分别点击两侧的「选择图片」，预览图片后点击「开始验证」。窗口会显示同人或异人的判断、余弦相似度及阈值；未检测到人脸时单独提示，文件无法读取或模型配置错误时显示异常信息。更换图片会清除旧结果，「清空图片」可重新开始。推理在后台线程执行，窗口保持响应。

GUI 使用与 `FaceVerification` 相同的模型、检测器和阈值，需在桌面环境或启用 WSLg 的 WSL 中运行。GUI 依赖 `PySide6-Essentials` 已加入 `requirements.txt`；已有环境可以单独执行 `python -m pip install PySide6-Essentials==6.8.3` 安装。支持指定权重和 CPU 线程数：

```bash
python gui.py --checkpoint 18.pth --threads 4
```

### 接口与命令行

仓库提供基线模型权重 `checkpoints/18.pth`。自行训练前，配置默认使用该权重；训练生成 `best.pth` 后，配置会优先使用后者。使用 CASIA 验证集标定基线模型的验证阈值：

```bash
python calibrate.py --checkpoint 18.pth --mode both
```

标定使用已有模型权重，不会训练或更新权重。程序生成两份文件：`checkpoints/validation_threshold.json` 用于已对齐图片，`checkpoints/verification_threshold.json` 用于包含检测和对齐步骤的接口。前一种流程使用全部 21,150 对 CASIA 验证样本；后一种流程还会执行 MTCNN 检测。使用 CPU 标定可能需要数分钟。

调用作业要求的接口：

```python
from face_verification import FaceVerification

y = FaceVerification("/absolute/path/face1.jpg", "/absolute/path/face2.jpg")
assert y in (0, 1)
```

也可以直接通过命令行运行：

```bash
python face_verification.py /absolute/path/face1.jpg /absolute/path/face2.jpg
```

模型和 MTCNN 检测器只加载一次，后续调用会复用它们。文件不存在或无法读取时，程序会抛出包含说明信息的异常。如果任意一张可正常读取的图片未检测到人脸，则返回 `0`。

`face_verification.py` 支持 `--checkpoint` 指定权重文件名（相对于 `checkpoints/`），支持 `--threads` 指定 PyTorch 的 CPU 线程数，默认值为 `4`。例如：

```bash
python face_verification.py face1.jpg face2.jpg --checkpoint 18.pth --threads 4
```

输出 `1` 表示同一个人，输出 `0` 表示不同的人。当前已经提供与 `18.pth` 对应的两份阈值文件，可以直接验证；更换模型后需要重新标定。标定会覆盖这两份阈值文件，运行时会检查阈值是否与指定权重匹配。

## 训练与评估

程序按照 `train.txt` 训练，使用 `val.txt` 验证。每轮训练结束后，根据验证集中的身份标签构造数量相同的同人和异人图片对，以配对准确率选择余弦相似度阈值，并将验证表现最好的网络保存为 `checkpoints/best.pth`，同时保存其对应的已对齐图片阈值。训练只使用 `train.txt` 列出的图片。

```bash
python train.py
python calibrate.py --checkpoint best.pth --mode both
```

重启 Python 后，接口会自动优先使用 `best.pth`。LFW 评估固定使用 CASIA 验证集标定的阈值，不会根据测试图片对重新选择阈值：

```bash
python test.py --checkpoint best.pth
```

默认评估使用 `lfw_funeled_cropped` 中的裁剪图片进行特征比较。如果需要在尺寸较大的已对齐源图片上评估包含检测、对齐步骤的完整流程，请运行：

```bash
python test.py --checkpoint best.pth --detect \
  --test-root datasets/scikit_learn_data/lfw_home/lfw_funneled
```

这里得到的是固定阈值下的测试结果，不是标准 LFW 十折交叉验证成绩。

`train.py` 的训练参数在 `config.py` 中设置。`calibrate.py` 的 `--mode` 可取 `aligned`（已对齐图片）、`detected`（检测与对齐流程）或 `both`（两种流程，默认值）。`test.py` 支持 `--pairs` 指定配对文件、`--test-root` 指定测试图片根目录、`--detect` 启用检测与对齐。标定和测试脚本也支持 `--checkpoint` 与 `--threads`。查看完整帮助：

```bash
python face_verification.py --help
python calibrate.py --help
python test.py --help
```

CASIA 训练集与验证集的图片不重复，但大部分人物身份重合。检测流程的阈值标定会排除包含未检出人脸图片的配对，并记录数量；完整流程的 LFW 测试保留所有配对，检测失败时预测为 `0`。项目没有附带 `18.pth` 的原始训练记录，因此其标定与 LFW 成绩仅用于检查基线效果，无法证明原始训练未使用验证或测试图片。完成作业时，应自行训练或微调模型，并说明模型来源。

## 实现验证

```bash
python -m unittest discover -s tests -v
```
