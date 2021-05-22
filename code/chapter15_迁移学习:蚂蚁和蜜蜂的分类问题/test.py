import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# 图片预处理
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


# 制作数据集
test_dataset = datasets.ImageFolder(
    root='./test',
    transform=test_transforms
)


# 数据加载器
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

# 加载模型
device = torch.device('cpu')
class_names = test_dataset.classes
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('model.pt', map_location=device))

# 可视化函数
def visualize_model(model):
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(test_loader))

        outputs = model(inputs)
        preds = outputs.argmax(1)

        plt.figure(figsize=(9, 9))
        for i in range(inputs.size(0)):
            plt.subplot(2, 2, i+1)
            plt.axis('off')
            plt.title(f'pred: {class_names[preds[i]]}|true: {class_names[labels[i]]}')
            im = inputs[i].permute(1, 2, 0)
            plt.imshow(im)
        plt.savefig('old.jpg')
        plt.show()

# 可视化结果
visualize_model(model)
