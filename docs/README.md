## 一种结合了代码、数据集和HTML的数字学习媒介
<br>

<table align="center">
<tr>
<td>

<img src="https://raw.githubusercontent.com/Charmve/computer-vision-in-action/main/res/ui/maiwei.png" align="left" alt="L0CV" width="120" title="有疑问，跑起来就会变成一朵花 ❀"/>
<code>全面</code>&nbsp;<code>前沿</code>&nbsp;<code>免费</code>
<h1> 计算机视觉实战演练：算法与应用 <sup> 📌</sup>
<br><em>Computer Vision in Action</em></h1>

<br>
<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/作者-@Charmve-000000.svg?logo=GitHub" alt="GitHub"></a>
  <a href="https://github.com/Charmve/computer-vision-in-action"><img src="https://img.shields.io/badge/CV-Action-yellow" alt="CV-Action"></a>
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

### 全书组织
🏷️ `fig_book_org`

<p align="center">
  <img src="https://raw.githubusercontent.com/Charmve/computer-vision-in-action/main/res/ui/book_org.png" width=80% alt="book_org.png">
</p> 

<p align="right">
  <a href="../res/计算机视觉实战演练：算法与应用_思维导图.pdf">全书详细思维导图</a>
</p>

本书详细介绍，请移步 [<b>序言</b>](book_preface.md)。

* 第一部分包括基础知识和预备知识。chap_introduction 提供深度学习的入门课程。然后在 chap_preliminaries 中，将快速向你介绍实践计算机视觉所需的前提条件，例如如何存储和处理数据，以及如何应用基于线性代数、微积分和概率基本概念的各种数值运算，涵盖了深度学习的最基本概念和技术，例如线性回归、多层感知机和正则化。

* 第二部分是本书涉及的计算机视觉基础理论，核心部分为神经网络模型。包括神经网络、卷积神经网络、循环神经网络理论讲解，以图像分类、模型拟合与优化作为其代码的实战项目。在模型拟合和优化章节中，着重分享梯度下降、随机梯度下降、动量法、AdaBoost等方法。

* 接下来的七章集中讨论现代计算机视觉技术实战。描述了计算机视觉的经典卷积神经网络PyTorch实现，并为我们随后实现更复杂的模型奠定了基础。接下来，在`chap_cnn` 和 `chap_modern_cnn` 中，我们介绍了卷积神经网络（convolutional neural network，CNN），这是构成大多数现代计算机视觉系统骨干的强大工具。随后，在 `chap_rnn` 和 `chap_modern_rnn` 中，我们引入了循环神经网络(recurrent neural network，RNN)，这是一种利用数据中的时间或序列结构的模型，通常用于自然语言处理和时间序列预测，但在计算机视觉领域也表现出惊奇的效果。在`chap_attention` 中，我们介绍了一类新的模型，它采用了一种称为注意力机制的技术，最近它们已经开始在自然语言处理中取代循环神经网络。这一部分将帮助你快速了解大多数现代计算机视觉应用背后的基本工具。

* 该部分以项目为实战指导，给出详细的项目指导书和代码实现，更为特别的是，给出了**notebook**可以直接在线运行，跑通结果，免去了本地运行环境的搭建复杂性。于此同时，为了方便读者在本地调试，作者建立了一个名为 L0CV 的第三方包，可以直接在代码中 ``import L0CV`` 后使用。

* 第三部分讨论最近几年出现的“网红”模型，诸如：Transformer、Attention、知识蒸馏、迁移学习、生成对抗模型等。这部分也是此份资料的力挺之作。首先，在 `chap_optimization` 中，我们讨论了用于训练深度学习模型的几种常用优化算法。

### 部分章节还在完善中，谢谢支持！

## 愿景

本开源项目代表了我们的一种尝试：我们将教给读者概念、背景知识和代码；我们将在同一个地方阐述剖析问题所需的批判性思维、解决问题所需的数学知识，以及实现解决方案所需的工程技能。

我们的目标是创建一个为实现以下目标的统一资源：
1. 所有人均可在网上免费获取；
2. 提供足够的技术深度，从而帮助读者实际成为计算机视觉应用科学家：既理解数学原理，又能够实现并不断改进方法；
3. 包含可运行的代码，为读者展示如何在实际中解决问题。这样不仅直接将数学公式对应成实际代码，而且可以修改代码、观察结果并及时获取经验；
4. 允许我们和整个[社区](https://github.com/Charmve/computer-vision-in-action/discussions)不断快速迭代内容，从而紧跟仍在高速发展的计算机视觉领域；
5. 由包含有关技术细节问答的论坛作为补充，使大家可以相互答疑并交换经验。


## 常见问题

- **在线教程页面无法打开**: 

    测试中存在部分人打不开在线教程的情况。

    部分小伙伴反馈尝试切换浏览器后可以正常打开了，如果仍然不行，最有效的解决办法是科学上网。

- **无法加载图片的解决办法**: 

    根本解决办法还是科学上网，也可以尝试修改host文件看下是否能解决。

    [windows解决方案：修改host文件](https://blog.csdn.net/u011583927/article/details/104384169)

- **公式无法正常显示解决办法**：

    GitHub中的Markdown原生是不支持LATEX公式显示的，如果你喜欢在本项目中直接浏览教程，可以安装Chrome的`MathJax Plugin for Github`插件让大部分公式正常显示。而docs文件夹已经利用docsify被部署到了GitHub Pages上，包含公式的章节强力建议使用 [《计算机视觉实战演练：算法与应用》 在线阅读](https://charmve.github.io/computer-vision-in-action) 进行学习。
    
    当然如果你还想跑一下运行相关代码的话还是得把本项目clone下来，然后运行code文件夹下相关代码。

- **Jupyter Notebook 无法在 GitHub 上呈现？** [使用 nbviewer](https://leaherb.com/notebook_wont_render_use_nbviewer/)。

## 致谢

<a href="https://maiweiai.github.io/"><img src="https://raw.githubusercontent.com/Charmve/computer-vision-in-action/main/res/ui/maiwei_ai.png" height="36" alt="迈微AI研习社" title="迈微AI研习社"> </a> <a href="https://madewithml.com/"><img src="https://madewithml.com/static/images/logo.png" height="30" alt="Made With ML" title="Made With ML"> </a> &nbsp;&nbsp; <a href="https://www.epubit.com/"><img src="https://cdn.ptpress.cn/pubcloud/3/app/0718A6B0/cover/20191204BD54009A.png" height="30" alt="异步社区" title="异步社区"> </a>  &nbsp;&nbsp; <a href="https://360.cn"><img src="https://p3.ssl.qhimg.com/t011e94f0b9ed8e66b0.png" height="36" alt="奇虎360" title="奇虎360"> </a>

## 参考文献

感谢前人的杰出工作，我才得以写出此书。点击[<b>这里</b>](REFERENCE.md)，查看全部参考文献列表

## 关注我们
<div align=center>
<p>扫描下方二维码，然后回复关键词“<b>计算机视觉实战教程</b>”，即可加入“读者交流群”</p>
<img src="https://user-images.githubusercontent.com/29084184/116501908-a63da600-a8e4-11eb-827c-7772655e0079.png" width = "250" height = "270" alt="迈微AI研习社是一个专注AI领域的开源组织，作者系CSDN博客专家，主要分享机器学习算法、计算机视觉等相关内容，每周研读顶会论文，持续关注前沿技术动态。底部有菜单分类，关注我们，一起学习成长。">
</div>

- 若本书里没有你想要理论和实战项目，或者你发现本书哪个地方有错误，请毫不犹豫地去本仓库的 Issues（ 地址 https://github.com/charmve/computer-vision-in-action/issues ）进行反馈，在对应版块提交你希望补充的内容或者勘误信息，作者通常会在 24 小时以内给您回复，超过 24 小时未回复的话可以邮件联系我（微信 MaiweiE_com）；
- 同时，我也欢迎大家加入本项目的建设中来，欢迎 [pull request](https://github.com/charmve/computer-vision-in-action/pulls)！

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"> 知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a> 进行许可。

## Citation

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
