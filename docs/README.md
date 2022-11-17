

<table align="center">
<tr>
<td>

<img src="https://raw.githubusercontent.com/Charmve/computer-vision-in-action/main/res/ui/maiwei.png" align="left" alt="L0CV" width="120" title="有疑问，跑起来就会变成一朵花 ❀"/>
<code>全面</code>&nbsp;<code>前沿</code>&nbsp;<code>免费</code>
<h1> 计算机视觉实战演练：算法与应用 <sup> 📌</sup>
<br><em>Computer Vision in Action</em></h1>

<br>

欢迎关注我的公众号 <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>

<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/作者-@Charmve-000000.svg?logo=GitHub" alt="GitHub"></a>
  <a href="https://github.com/Charmve/computer-vision-in-action/" target="_blank"><img src="https://img.shields.io/badge/-💮 %20L0CV-lightgreen.svg" alt="L0CV" title="L0CV"></a>
  <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>
  <a href="https://github.com/Charmve/computer-vision-in-action/edit/master/README.md"><img src="https://img.shields.io/github/stars/Charmve/computer-vision-in-action?style=social" alt="Stars"></a>
  <a href="https://github.com/Charmve/computer-vision-in-action/edit/master/README.md"><img src="https://img.shields.io/github/forks/Charmve/computer-vision-in-action?style=social" alt="Forks"></a>
</p>

<div align="center">
	<img src="https://raw.githubusercontent.com/Charmve/computer-vision-in-action/main/res/ui/L0CV.png" width=60% alt="L0CV architecture">
</div>

</tr>
</td>
</table>

<br>

> <b>"如果你只是看了这个项目的在线文档，那么你并没有利用好这个项目。太可惜！"</b>

<br>

## ✨ Features

- 一种结合了代码、图示和HTML的在线学习媒介
- 跨平台，只需一个浏览器即可！
- "以用促学，先会后懂"

## 📕 全书组织
🏷️ `fig_book_org`

<p align="center">
  <img src="https://raw.githubusercontent.com/Charmve/computer-vision-in-action/main/res/ui/book_org.png" width=90% alt="book_org.png">
</p> 

<p align="right">
  <a href="../res/计算机视觉实战演练：算法与应用_思维导图.pdf">全书详细思维导图</a>
</p>

本书详细介绍，请移步 [<b>序言</b>](book_preface.md)。

* 第一部分包括基础知识和预备知识。提供深度学习的入门课程，然后在理论篇中，将快速向你介绍实践计算机视觉所需的前提条件，例如如何存储和处理数据，以及如何应用基于线性代数、微积分和概率基本概念的各种数值运算，涵盖了深度学习的最基本概念和技术，例如线性回归、多层感知机和正则化。

* 第二部分是本书涉及的计算机视觉基础理论，核心部分为神经网络模型，包括神经网络、卷积神经网络、循环神经网络理论讲解，以图像分类、模型拟合与优化作为其代码的实战项目。在模型拟合和优化章节中，着重分享梯度下降、随机梯度下降、动量法、AdaBoost等方法。

* 接下来的七章集中讨论现代计算机视觉技术实战，也是本书的核心部分。围绕这样的组织逻辑：什么是计算机视觉？计算机视觉解决什么问题，都是怎么解决的？传统方法——以卷积神经网络为中心的神经网络；现代方法——Transformer、强化学习、迁移学习、生成对抗等。各种方法是如何实现的，用到了什么框架？在第7章中，描述了计算机视觉的经典卷积神经网络PyTorch实现，并为我们随后实现更复杂的模型奠定了基础。在随后的几个章节中，我们主要解决图像分类、目标检测、语义分割、3D重建等实际问题，并给出实战项目。

* 该部分以项目为实战指导，给出详细的项目指导书和代码实现，更为特别的是，给出了**notebook**可以直接在线运行，跑通结果，免去了本地运行环境的搭建复杂性。于此同时，为了方便读者在本地调试，作者建立了一个名为 ``L0CV`` 的第三方包，可以直接在代码中 ``import L0CV`` 后使用。

* 第三部分讨论最近几年出现的<b>“网红”模型</b>，诸如：Transformer、Attention、知识蒸馏、迁移学习、生成对抗模型等。这部分也是此份资料的力挺之作。最后，在 `chap_optimization` 中，我们讨论了用于训练深度学习模型的几种常用优化算法，如：模型压缩、模型剪枝、微调、蒸馏等。

### 部分章节还在完善中，谢谢支持！

## 🌈 愿景

本开源项目代表了我们的一种尝试：我们将教给读者概念、背景知识和代码；我们将在同一个地方阐述剖析问题所需的批判性思维、解决问题所需的数学知识，以及实现解决方案所需的工程技能。

我们的目标是创建一个为实现以下目标的统一资源：
1. 所有人均可在网上免费获取；
2. 提供足够的技术深度，从而帮助读者实际成为计算机视觉应用科学家：既理解数学原理，又能够实现并不断改进方法；
3. 包含可运行的代码，为读者展示如何在实际中解决问题。这样不仅直接将数学公式对应成实际代码，而且可以修改代码、观察结果并及时获取经验；
4. 允许我们和整个[社区](https://github.com/Charmve/computer-vision-in-action/discussions)不断快速迭代内容，从而紧跟仍在高速发展的计算机视觉领域；
5. 由包含有关技术细节问答的论坛作为补充，使大家可以相互答疑并交换经验。

## L0CV DemoDay

<p align="center">
  <img src="https://raw.githubusercontent.com/Charmve/computer-vision-in-action/main/res/ui/demoday.png" title="L0CV Demo Day"> <a href="https://github.com/Charmve/computer-vision-in-action/tree/main/L0CV-Universe">L0CV-Universe</a>
</p>

如果你也是从这里出发，在开源的项目中应用进去，并在标题下给出引用 <a href="https://github.com/Charmve/computer-vision-in-action/" target="_blank"><img src="https://img.shields.io/badge/-💮 %20L0CV-lightgreen.svg" alt="L0CV" title="L0CV"></a>，您的项目将会在这里展现！

<h5 align="center"><i>以用促学，先会后懂。理解深度学习的最佳方法是学以致用。</i></h5>

<table class="table table-striped table-bordered table-vcenter">
    <tbody class=ai-notebooks-table-content>
    <tr>
        <td>
            <div class="mdl-cell mdl-cell--4-col">
                <a href="https://nbviewer.jupyter.org/format/slides/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter02_CNN/Pytorch_MNIST.ipynb" target="_blank"><img width="%40"  src="https://user-images.githubusercontent.com/29084184/128386334-e0273125-5d51-4e33-a6e7-f2e732fb0836.png"></a>
            </div>
        </td>
        <td>
            <div class="mdl-cell mdl-cell--4-col">
                <a href="https://github.com/Charmve/computer-vision-in-action/blob/main/docs/2__实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/7.12%20实战Kaggle比赛：图像分类（CIFAR-10）.md" target="_blank"><img width="%40"  src="https://user-images.githubusercontent.com/29084184/128386363-dc0c987c-b374-4e43-9e56-65c30f7a1899.png"></a>
            </div>
        </td>
        <td>
            <div class="mdl-cell mdl-cell--4-col">
                <a href="https://github.com/Charmve/computer-vision-in-action/blob/main/docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/7.13%20实战Kaggle比赛：狗的品种识别（ImageNet%20Dogs）.md" target="_blank"><img width="%40"  src="https://user-images.githubusercontent.com/29084184/128386468-ca555572-a98d-44c5-bdef-442371322ee7.png"></a>
            </div>
        </td>
    </tr>
    <tr>
        <td>
            <div class="mdl-cell mdl-cell--4-col">
                <a href="https://nbviewer.jupyter.org/format/slides/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter12_practice-projects/Bringing-Old-Photo-Back-to-Life.ipynb" target="_blank"><img width="%40"  src="https://user-images.githubusercontent.com/29084184/128386504-25a9f798-be68-430d-b771-bd8db607c151.png"></a>
                </div>
        </td>
        <td>
            <div class="mdl-cell mdl-cell--4-col">
                <a href="https://github.com/Charmve/Awesome-Lane-Detection/tree/main/lane-detector" target="_blank"><img width="%40"  src="https://user-images.githubusercontent.com/29084184/128386530-d64210f0-d903-4004-9f6c-eb480d326241.png"></a>
                </div>
        </td>
        <td>
            <div class="mdl-cell mdl-cell--4-col">
                <a href="https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_进阶篇/chapter12-生成对抗模型/chapter12.3.3_neural-style.md" target="_blank"><img width="%40"  src="https://user-images.githubusercontent.com/29084184/128386405-4223b171-a318-4f76-93b3-0fff016aa39f.png"></a>
                </div>
        </td>
    </tr>
    </tbody>
</table>


*《计算机视觉实战演练：算法与应用》V1.2 *部分项目还在更新中*

<br>
<p align="left">
  <a href="https://github.com/Charmve/computer-vision-in-action"><img src="https://raw.githubusercontent.com/Charmve/computer-vision-in-action/main/res/ui/workswith1.png" title="Works with L0CV" width="120"></a>
</p>

| 实战项目 | 章节 | Binder | Google Colab | 
| :-- | :---| :---:| :---: |
| [实战项目 1 - 手写字分类](https://blog.csdn.net/Charmve/article/details/108531735) | 第 1 章 - 神经网络 | | |
| [实战项目 2 - 动手搭建一个卷积神经网络](/docs/1_理论篇/chapter2_CNN/chapter2_CNN-in-Action.md) | 第 2 章 - 卷积神经网络 | | |
| [实战项目 3 - 基于卷积神经网络的人脸表情识别](https://blog.csdn.net/charmve/category_9754344.html) | 第 3 章 - 图像分类 | | |
| [实战项目 4 - 使用卷积神经网络对CIFAR10图片进行分类](http://mp.weixin.qq.com/s?__biz=MzIxMjg1Njc3Mw%3D%3D&chksm=97bef597a0c97c813e185e1bbf987b93d496c6ead8371364fd175d9bac46e6dcf7059cf81cb2&idx=1&mid=2247487293&scene=21&sn=89684d1c107177983dc1b4dca8c20a5b#wechat_redirect) | 第 3 章 - 图像分类 | | |
| [实战项目 5 - 使用OpenCV进行图像全景拼接](https://blog.csdn.net/Charmve/article/details/107897468) | 第 6 章 - 软件环境搭建与工具使用 | <a href="https://nbviewer.jupyter.org/format/slides/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter08_environment-setup-and-tool-use/OpenCV-ImageStitching.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a> | <a target="_blank" href="https://colab.research.google.com/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter08_environment-setup-and-tool-use/OpenCV-ImageStitching.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" ></a> |
| [实战项目 6 - Kaggle比赛：图像分类（CIFAR-10）](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/7.12%20实战Kaggle比赛：图像分类（CIFAR-10）.md) | 第 8 章 - 著名数据集及基准 | <a href="https://nbviewer.jupyter.org/format/slides/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter10_dataset-and-benchmark/kaggle_cifar10.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a> | <a target="_blank" href="https://colab.research.google.com/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter10_dataset-and-benchmark/kaggle_cifar10.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" ></a> |
| [实战项目 7 - Kaggle比赛：狗的品种识别（ImageNet Dogs）](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/7.13%20实战Kaggle比赛：狗的品种识别（ImageNet%20Dogs）.md) | 第 8 章 - 著名数据集及基准 | <a href="https://nbviewer.jupyter.org/format/slides/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter10_dataset-and-benchmark/kaggle_dog.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a> | <a target="_blank" href="https://colab.research.google.com/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter10_dataset-and-benchmark/kaggle_dog.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" ></a> |
| [实战项目 8 - 基于PolarNet的点云端到端语义分割项目实战]() | 第 9 章 - 检测与分割实战项目 || |
| [实战项目 9 - 基于PyTorch的YOLO5目标检测项目实战]() | 第 9 章 - 检测与分割实战项目 || |
| [实战项目 10 - 实时高分辨率背景抠图](/docs/2_实战篇/chapter9_检测与分割实战项目/9.3%20实例分割.md#932-实战项目-8-实时高分辨率背景抠图) | 第 9 章 - 检测与分割实战项目 || |
| [实战项目 11 - 车道线检测项目实战](https://blog.csdn.net/Charmve/article/details/116678477) | 第 10 章 - 计算机视觉课题研究初探 | | |
| [实战项目 12 - PyTorch 如何使用TensorBoard](/docs/3_进阶篇/chapter11-可视化和理解/chapter11-可视化和理解.md) | 第 13 章 - 可视化和理解 | | |
| [实战项目 13 - 图像样式迁移]() | 第 14 章 生成对抗模型 | <a href="https://nbviewer.jupyter.org/format/slides/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter09_computer-vision/9.11_neural-style.ipynb#/"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a> | <a target="_blank" href="https://colab.research.google.com/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter09_computer-vision/9.11_neural-style.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" ></a> |
| [实战项目 14 - 旧照片修复](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life) | 第 14 章 - 生成对抗模型 | <a href="https://nbviewer.jupyter.org/format/slides/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter12_practice-projects/Bringing-Old-Photo-Back-to-Life.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a> | <a target="_blank" href="https://colab.research.google.com/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter09_computer-vision/9.11_neural-style.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" ></a> |
| [实战项目 15 - 动漫头像生成](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life) | 第 14 章 - 生成对抗模型 | <a href="https://nbviewer.jupyter.org/format/slides/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter12_practice-projects/Anime-StyleGAN2.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a> | <a target="_blank" href="https://colab.research.google.com/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter09_computer-vision/9.11_neural-style.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" ></a> |
| [项目实战 16 - 视频理解项目实战 SlowFast + Multi-Moments in Time](https://charmve.github.io/computer-vision-in-action/#/3_进阶篇/chapter14-视频理解/chapter14-视频理解?id=_147-视频理解项目实战) | 第 16 章 - 视频理解 | <a href="https://nbviewer.jupyter.org/format/slides/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter12_practice-projects/Bringing-Old-Photo-Back-to-Life.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a> | <a target="_blank" href="https://colab.research.google.com/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter09_computer-vision/9.11_neural-style.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" ></a> |
| [实战项目 17 - 蚂蚁和蜜蜂的分类问题](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/3_进阶篇/chapter15_迁移学习/chapter15_迁移学习的应用.md) | 第 17 章 - 迁移学习 | <a href="https://nbviewer.jupyter.org/format/slides/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter15_Transfer-Learning/TL-ants-bees-classification.ipynb#/" /> <img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a> | <a target="_blank" href="https://colab.research.google.com/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter09_computer-vision/9.11_neural-style.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" ></a> |
| [实战项目 18 - 基于Transformer的视频实例分割网络VisTR (CVPR2021)](https://blog.csdn.net/Charmve/article/details/115339803) | 第 19 章 - 跨界模型 Transformer | | |
| [实战项目 19 - 支付宝CVPR细粒度视觉分类挑战赛夺冠方案解读](https://mp.weixin.qq.com/s/RTkBQJ7Uj86Wxt7HmwWKzA)| 第 20 章 - 知识蒸馏 | | |
| ...  | ... | ... |

<br>

## 🔎 如何食用

🏷️ `sec_code`

<details><summary>详细攻略展开</summary>

### 方式一 Jupyter Notebook (推荐方式 ✨)

#### 1. 本地运行

- 依赖包安装
```
pip3 install -r requirements.txt
```
- 安装 Jupyter
```
python3 -m pip install --upgrade pip
python3 -m pip install jupyter
```
- 查看并运行jupyter

请在终端（Mac / Linux）或命令提示符（Windows）上运行以下命令：

```shell
cd notebooks
jupyter notesbook
```

#### 2. 远程运行

- 打开每章节首页，点击 <a target="_blank" href="https://colab.research.google.com/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter09_computer-vision/9.11_neural-style.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" ></a> 可直接打开 Google Colab ，点击 <code><img height="20" src="https://user-images.githubusercontent.com/29084184/126463073-90077dff-fb7a-42d3-af6b-63c357d6db9f.png" alt="Copy to Drive" title="Copy to Drive"></code> [Copy to Drive] 即可在线运行测试。 

- 点击 <a href="https://mybinder.org/v2/gh/Charmve/computer-vision-in-action/main/notebooks/"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a> 也可在 ``mybinder`` 查看和在线运行。

<p align="center">
  <img src="https://user-images.githubusercontent.com/29084184/126031057-1e6ca67f-4475-47c1-a6ff-66375cb86908.png" width=60% alt="Run on Colab" title="Run on Colab">
  <br>
  图2 例子：12.3.3 样式迁移
</p> 

<p align="center">
  <img src="https://user-images.githubusercontent.com/29084184/126031137-14e349cd-1e89-4f98-9c56-0f1d3007ed89.png" width=60% alt="点击 Copy to Drive">
  <br>图3 例子：12.3.3 样式迁移 Colab 点击 <code><img height="20" src="https://user-images.githubusercontent.com/29084184/126463073-90077dff-fb7a-42d3-af6b-63c357d6db9f.png" alt="Copy to Drive" title="Copy to Drive"></code> [Copy to Drive]
</p>

### 方式二 使用 ``/code`` 
#### 1. 运行环境 + L0CV 加载
- 依赖包安装
```shell
sudo apt-get update
pip3 install -r requirements.txt
```
- 创建 L0CV
```shell
python3 setup.py
```
- 测试环境

```shell
cd code
python3 L0CV_test.py
```

#### 2. 直接调用每个章节的代码测试

```python3
import L0CV
```

[收起](#🔎-如何食用)
																  
</details>

<br>

## 📢 常见问题

- **在线教程页面无法打开**：

    测试中存在部分人打不开在线教程的情况。

    部分小伙伴反馈尝试切换浏览器后可以正常打开了，如果仍然不行，最有效的解决办法是科学上网。

- **无法加载图片的解决办法**：

    根本解决办法还是科学上网，也可以尝试修改host文件看下是否能解决。

    解决方案: 修改host文件 <code><a href="https://www.jianshu.com/p/25e5e07b2464"><img height="20" src="https://user-images.githubusercontent.com/29084184/126457822-d431fb90-6b9e-4a4e-bedc-3c598e9e2ee2.png" alt="Apple" title="Apple"></code> Mac</a> <code><a href="https://blog.csdn.net/u011583927/article/details/104384169"><img height="20" src="https://user-images.githubusercontent.com/29084184/126457902-0c1a71c2-f920-45a1-a143-ce8b5c435fe7.png" alt="Win10" title="Win10"></code> Windows</a>

- **公式无法正常显示解决办法**：

    GitHub中的Markdown原生是不支持LATEX公式显示的，如果你喜欢在本项目中直接浏览教程，可以安装Chrome的`MathJax Plugin for Github`插件让大部分公式正常显示。而docs文件夹已经利用docsify被部署到了GitHub Pages上，包含公式的章节强力建议使用 [《计算机视觉实战演练：算法与应用》 在线阅读](https://charmve.github.io/computer-vision-in-action) 进行学习。
    
    当然如果你还想跑一下运行相关代码的话还是得把本项目clone下来，然后运行code文件夹下相关代码。

- **Jupyter Notebook 无法在 GitHub 上呈现？** [使用 nbviewer](https://leaherb.com/notebook_wont_render_use_nbviewer/)。 <a href="https://mybinder.org/v2/gh/Charmve/computer-vision-in-action/main/notebooks/"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>


## 💬 Community

- <b>We have a discord server!</b> [![Discord](https://img.shields.io/discord/744385009028431943.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/9fTPvAY2TY) <em>This should be your first stop to talk with other friends in ACTION. Why don't you introduce yourself right now? [Join the ACTION channel in L0CV Discord](https://discord.gg/9fTPvAY2TY)</em>

- <b>L0CV-微信读者交流群</b> <em>关注公众号迈微AI研习社，然后回复关键词“<b>计算机视觉实战教程</b>”，即可加入“读者交流群”</p></em>

## 🛡 LICENSE

<a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank" style="display:inline-block"><img src="https://img.shields.io/badge/license-Apache%202.0-red?logo=apache" alt="Code License"></a> <a rel="DocLicense" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/docs%20license-CC%20BY--NC--SA%204.0-green?logo=creativecommons" title="CC BY--NC--SA 4.0"/></a>
	
- ``L0CV``代码部分采用 [Apache 2.0协议](https://www.apache.org/licenses/LICENSE-2.0) 进行许可，包括名为 <b><em>L0CV</em></b> 的原创第三方库、``/code``和``/notebook``下的源代码。遵循许可的前提下，你可以自由地对代码进行修改，再发布，可以将代码用作商业用途。但要求你：
  - **署名**：在原有代码和衍生代码中，保留原作者署名及代码来源信息。
  - **保留许可证**：在原有代码和衍生代码中，保留``Apache 2.0``协议文件。

- ``L0CV``文档部分采用 [知识共享署名 4.0 国际许可协议](http://creativecommons.org/licenses/by/4.0/) 进行许可。 遵循许可的前提下，你可以自由地共享，包括在任何媒介上以任何形式复制、发行本作品，亦可以自由地演绎、修改、转换或以本作品为基础进行二次创作。但要求你：
  - **署名**：应在使用本文档的全部或部分内容时候，注明原作者及来源信息。
  - **非商业性使用**：不得用于商业出版或其他任何带有商业性质的行为。如需商业使用，请联系作者。
  - **相同方式共享的条件**：在本文档基础上演绎、修改的作品，应当继续以知识共享署名 4.0国际许可协议进行许可。

## 👥 社区互助

如果您在使用的过程中碰到问题，可以通过下面几个途径寻求帮助，同时我们也鼓励资深用户通过下面的途径给新人提供帮助。

- 通过 <a href="https://github.com/Charmve/computer-vision-in-action/discussions" target="_blank" style="display:inline-block"><img src="https://img.shields.io/badge/GitHub-Discussions-green?logo=github" alt="GitHub Discuss"></a> 提问时，建议使用 `Q&A` 标签。

- 通过 <a href="http://stackoverflow.com/questions/tagged/L0CV" target="_blank" style="display:inline-block"><img src="https://img.shields.io/badge/-Stack%20Overflow-gray?logo=stackoverflow" alt="Stack Overflow"></a> 或者 <a href="https://segmentfault.com/t/L0CV" target="_blank" style="display:inline-block"><img src="https://img.shields.io/badge/-Segment%20Fault-gray?logo=mongodb" alt="Segment Fault"></a> 提问时，建议加上 `L0CV` 标签。

- <a href="https://segmentfault.com/t/L0CV" target="_blank" style="display:inline-block"><img src="https://img.shields.io/badge/微信-L0CV-green?logo=wechat" alt="Segment Fault"></a> 微信、知乎、微博开话题可以生成tag，如微信聊天、朋友圈加 ``#L0CV`` 可话题交流。

## 👐 关注我们
<div align=center>
<p>扫描下方二维码，然后回复关键词“<b>计算机视觉实战教程</b>”，即可加入“读者交流群”</p>
<img src="https://user-images.githubusercontent.com/29084184/116501908-a63da600-a8e4-11eb-827c-7772655e0079.png" width = "250" height = "270" alt="迈微AI研习社是一个专注AI领域的开源组织，作者系CSDN博客专家，主要分享机器学习算法、计算机视觉等相关内容，每周研读顶会论文，持续关注前沿技术动态。底部有菜单分类，关注我们，一起学习成长。">
</div>

- 若本书里没有你想要理论和实战项目，或者你发现本书哪个地方有错误，请毫不犹豫地去本仓库的 Issues（ 地址 https://github.com/charmve/computer-vision-in-action/issues ）进行反馈，在对应版块提交你希望补充的内容或者勘误信息，作者通常会在 24 小时以内给您回复，超过 24 小时未回复的话可以邮件联系我（微信 MaiweiE_com）；

- 同时，我也欢迎大家加入本项目的建设中来，欢迎 [pull request](https://github.com/charmve/computer-vision-in-action/pulls)！

- <em>请尽管表达你们的意见和建议，GitHub issues 和 电子书下方都可以留言，也可写邮件给我，我一定会回！</em>

## 💖 致谢

<a href="https://maiweiai.github.io/"><img src="https://raw.githubusercontent.com/Charmve/computer-vision-in-action/main/res/ui/maiwei_ai.png" height="36" alt="迈微AI研习社" title="迈微AI研习社"> </a> <a href="https://madewithml.com/"><img src="https://madewithml.com/static/images/logo.png" height="30" alt="Made With ML" title="Made With ML"> </a> &nbsp;&nbsp; <a href="https://www.epubit.com/"><img src="https://cdn.ptpress.cn/pubcloud/3/app/0718A6B0/cover/20191204BD54009A.png" height="30" alt="异步社区" title="异步社区"> </a>  &nbsp;&nbsp; <a href="https://360.cn"><img src="https://p3.ssl.qhimg.com/t011e94f0b9ed8e66b0.png" height="36" alt="奇虎360" title="奇虎360"> </a>

## 📎 参考文献

感谢前人的杰出工作，我才得以写出此书。点击[<b>这里</b>](REFERENCE.md)，查看全部参考文献列表。

## 📑 Citation

Use this bibtex to cite this repository:
```
@misc{computer-vision-in-action,
  title={计算机视觉实战演练：算法与应用（Computer Vision in Action）},
  author={Charmve},
  year={2021.06},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/Charmve/computer-vision-in-action}},
}
```

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/social-share.js/1.0.16/css/share.min.css">
<div class="social-share"></div>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/social-share.js/1.0.16/js/social-share.min.js"></script>
