# EfficientNet PyTorch

## Quickstart

Install with `pip install efficientnet_pytorch` and load a pretrained EfficientNet with:
```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')
```

* Source code: https://github.com/lukemelas/EfficientNet-PyTorch

### Table of contents
1. [About EfficientNet](#about-efficientnet)
2. [About EfficientNet-PyTorch](#about-efficientnet-pytorch)
3. [Installation](#installation)
4. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Classify](#example-classification)
    * [Example: Extract features](#example-feature-extraction)
    * [Example: Export to ONNX](#example-export)
6. [Contributing](#contributing)

## About EfficientNet

If you're new to EfficientNets, here is an explanation straight from the official TensorFlow implementation:

EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models. We develop EfficientNets based on AutoML and Compound Scaling. In particular, we first use [AutoML Mobile framework](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html) to develop a mobile-size baseline network, named as EfficientNet-B0; Then, we use the compound scaling method to scale up this baseline to obtain EfficientNet-B1 to B7.

<table border="0">
<tr>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/params.png" width="100%" />
    </td>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/flops.png", width="90%" />
    </td>
</tr>
</table>

EfficientNets achieve state-of-the-art accuracy on ImageNet with an order of magnitude better efficiency:


* In high-accuracy regime, our EfficientNet-B7 achieves state-of-the-art 84.4% top-1 / 97.1% top-5 accuracy on ImageNet with 66M parameters and 37B FLOPS, being 8.4x smaller and 6.1x faster on CPU inference than previous best [Gpipe](https://arxiv.org/abs/1811.06965).

* In middle-accuracy regime, our EfficientNet-B1 is 7.6x smaller and 5.7x faster on CPU inference than [ResNet-152](https://arxiv.org/abs/1512.03385), with similar ImageNet accuracy.

* Compared with the widely used [ResNet-50](https://arxiv.org/abs/1512.03385), our EfficientNet-B4 improves the top-1 accuracy from 76.3% of ResNet-50 to 82.6% (+6.3%), under similar FLOPS constraint.

### About EfficientNet PyTorch

EfficientNet PyTorch is a PyTorch re-implementation of EfficientNet. It is consistent with the [original TensorFlow implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), such that it is easy to load weights from a TensorFlow checkpoint. At the same time, we aim to make our PyTorch implementation as simple, flexible, and extensible as possible.

If you have any feature requests or questions, feel free to leave them as GitHub issues!

### Installation

Install via pip:
```bash
pip install efficientnet_pytorch
```

Or install from source:
```bash
git clone https://github.com/lukemelas/EfficientNet-PyTorch
cd EfficientNet-Pytorch
pip install -e .
```

### Usage

#### Loading pretrained models

Load an EfficientNet:
```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name('efficientnet-b0')
```

Load a pretrained EfficientNet:
```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')
```

Details about the models are below:

|    *Name*         |*# Params*|*Top-1 Acc.*|*Pretrained?*|
|:-----------------:|:--------:|:----------:|:-----------:|
| `efficientnet-b0` |   5.3M   |    76.3    |      ✓      |
| `efficientnet-b1` |   7.8M   |    78.8    |      ✓      |
| `efficientnet-b2` |   9.2M   |    79.8    |      ✓      |
| `efficientnet-b3` |    12M   |    81.1    |      ✓      |
| `efficientnet-b4` |    19M   |    82.6    |      ✓      |
| `efficientnet-b5` |    30M   |    83.3    |      ✓      |
| `efficientnet-b6` |    43M   |    84.0    |      ✓      |
| `efficientnet-b7` |    66M   |    84.4    |      ✓      |


#### Example: Classification

Below is a simple, complete example. It may also be found as a jupyter notebook in `examples/simple` or as a [Colab Notebook](https://colab.research.google.com/drive/1Jw28xZ1NJq4Cja4jLe6tJ6_F5lCzElb4).

We assume that in your current directory, there is a `img.jpg` file and a `labels_map.txt` file (ImageNet class names). These are both included in `examples/simple`.

```python
import json
from PIL import Image
import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')

# Preprocess image
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(Image.open('img.jpg')).unsqueeze(0)
print(img.shape) # torch.Size([1, 3, 224, 224])

# Load ImageNet class names
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img)

# Print predictions
print('-----')
for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
```

#### Example: Feature Extraction

You can easily extract features with `model.extract_features`:
```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')

# ... image preprocessing as in the classification example ...
print(img.shape) # torch.Size([1, 3, 224, 224])

features = model.extract_features(img)
print(features.shape) # torch.Size([1, 1280, 7, 7])
```

#### Example: Export to ONNX

Exporting to ONNX for deploying to production is now simple:
```python
import torch
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b1')
dummy_input = torch.randn(10, 3, 240, 240)

model.set_swish(memory_efficient=False)
torch.onnx.export(model, dummy_input, "test-b1.onnx", verbose=True)
```

[Here](https://colab.research.google.com/drive/1rOAEXeXHaA8uo3aG2YcFDHItlRJMV0VP) is a Colab example.


#### ImageNet

See `examples/imagenet` for details about evaluating on ImageNet.
