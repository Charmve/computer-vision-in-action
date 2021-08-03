<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

## 如何食用

:label:``sec_code``

<p align="center">
  <img src="../res/ui/L0CV.png" height="auto" width="60%" alt="L0CV architecture">
</p> 

### 1. 本地运行

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

### 2. 远程运行

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


### 3. 一种结合了代码、图示和HTML的在线学习媒介

按书中内容先后顺序逐章阅读，或者选取特定章节祥读 📁 <code>docs/</code> <sup>1</sup>，动手实践章节代码，在代码文件 📁 <code>code/</code> <sup>2</sup> 下找到对应代码，本地测试或者Colab 📁 <code>notebooks/</code> <sup>3</sup> 在线测试，方法如下面的示例。

## To-do 

| Scenario | Description |
| -------- | ----------- |
| - [ ] [Classification](classification) | Image Classification is a supervised machine learning technique to learn and predict the category of a given image. |
| - [ ] [Similarity](similarity)  | Image Similarity is a way to compute a similarity score given a pair of images. Given an image, it allows you to identify the most similar image in a given dataset.  |
| - [ ] [Detection](detection) | Object Detection is a technique that allows you to detect the bounding box of an object within an image. |
| - [ ] [Keypoints](keypoints) | Keypoint Detection can be used to detect specific points on an object. A pre-trained model is provided to detect body joints for human pose estimation. |
| - [ ] [Segmentation](segmentation) | Image Segmentation assigns a category to each pixel in an image. |
| - [ ] [Action Recognition](action_recognition) | Action Recognition (also known as activity recognition) consists of classifying various actions from a sequence of frames, such as "reading" or "drinking". |
| - [ ] [Tracking](tracking) | Tracking allows to detect and track multiple objects in a video sequence over time. |

*Cite from [<b><em>microsoft/computervision-recipes</em></b>](https://github.com/microsoft/computervision-recipes)

## 👥 Community

- <b>We have a discord server!</b> [![Discord](https://img.shields.io/discord/744385009028431943.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/9fTPvAY2TY) <em>This should be your first stop to talk with other friends in ACTION. Why don't you introduce yourself right now? [Join the ACTION channel in L0CV Discord](https://discord.gg/9fTPvAY2TY)</em>

- <b>L0CV-微信读者交流群</b> <em>关注公众号迈微AI研习社，然后回复关键词“<b>计算机视觉实战教程</b>”，即可加入“读者交流群”</p></em>

## LICENSE

<a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank" style="display:inline-block"><img src="https://img.shields.io/badge/license-Apache%202.0-red?logo=apache" alt="Code License"></a> <a rel="DocLicense" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/docs%20license-CC%20BY--NC--SA%204.0-green?logo=creativecommons" title="CC BY--NC--SA 4.0"/></a>
	
- ``L0CV``代码部分采用 [Apache 2.0协议](https://www.apache.org/licenses/LICENSE-2.0) 进行许可，包括名为 <b><em>L0CV</em></b> 的原创第三方库、``/code``和``/notebook``下的源代码。遵循许可的前提下，你可以自由地对代码进行修改，再发布，可以将代码用作商业用途。但要求你：
  - **署名**：在原有代码和衍生代码中，保留原作者署名及代码来源信息。
  - **保留许可证**：在原有代码和衍生代码中，保留``Apache 2.0``协议文件。

- ``L0CV``文档部分采用 [知识共享署名 4.0 国际许可协议](http://creativecommons.org/licenses/by/4.0/) 进行许可。 遵循许可的前提下，你可以自由地共享，包括在任何媒介上以任何形式复制、发行本作品，亦可以自由地演绎、修改、转换或以本作品为基础进行二次创作。但要求你：
  - **署名**：应在使用本文档的全部或部分内容时候，注明原作者及来源信息。
  - **非商业性使用**：不得用于商业出版或其他任何带有商业性质的行为。如需商业使用，请联系作者。
  - **相同方式共享的条件**：在本文档基础上演绎、修改的作品，应当继续以知识共享署名 4.0国际许可协议进行许可。
