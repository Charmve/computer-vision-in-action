<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

**第 10 章 计算机视觉课题研究初探**

作者: 张伟 (Charmve)

日期: 2021/04/29

目录

  - 第 10 章 [图像分类项目实战](https://charmve.github.io/computer-vision-in-action/#/chapter10/chapter10)
    - 10.1 [手写字识别](https://blog.csdn.net/Charmve/article/details/108531735)
    - 10.2 [文本检测](https://github.com/Charmve/Scene-Text-Detection)
    - 10.3 [车道线检测](https://github.com/Charmve/Awesome-Lane-Detection)
      - 10.3.1 常用网络 
      - 10.3.2 [实战项目 9 - 车道线检测项目实战](https://blog.csdn.net/Charmve/article/details/116678477)
    - 10.4 [镜面检测](https://github.com/Charmve/Mirror-Glass-Detection)
    - 10.5 [图像抠图/Matting](https://github.com/Charmve/TimeWarp)
    - 10.6 [图像超分辨率](charpter10_6-图像超分辨率.md)
    - 10.7 [3D 重建](charpter10_7-3D重建.md)
    - 小结
    - 参考文献

---


# 10.5 图像抠图



<table>
<tr>
	<td><font size="4">1.</font></td>
	<td><center><a href="https://www.youtube.com/watch?v=yvNZGDZC3F8" target="_blank"><img src="https://img-blog.csdnimg.cn/img_convert/53252d7269b7a184bab9824d5e039787.png" width="820" height="%100"/></a></center>
	</td>
	<td>
		<p align="center"><font size="5"><b>PointRend</b>: Image Segmentation as Rendering</font>
			<br><b>Alexander Kirillov<sup>∗</sup></b>, Yuxin Wu, Kaiming He, Ross Girshick
		</p>
		<p align="left">
	        <font size =2> <b>Abstract</b>. We present a new method for efficient high-quality image segmentation of objects and scenes. By analogizing classical computer graphics methods for efficient rendering with over- and undersampling challenges faced in pixel labeling tasks, we develop a unique perspective of image segmentation as a rendering problem. From this vantage, we present the PointRend (Point-based Rendering) neural network module: a module that performs point-based segmentation predictions
		<details>
  		    <summary>Click to expand</summary>
			at adaptively selected locations based on an iterative subdivision algorithm. PointRend can be flexibly applied to both instance and semantic segmentation tasks by building on top of existing state-of-the-art models. While many concrete implementations of the general idea are possible, we show that a simple design already achieves excellent results. Qualitatively, PointRend outputs crisp object boundaries in regions that are over-smoothed by previous methods. Quantitatively, PointRend yields significant gains on COCO and Cityscapes, for both instance and semantic segmentation. PointRend's efficiency enables output resolutions that are otherwise impractical in terms of memory or computation compared to existing approaches. 
		</details>
	        </font>
	     	</p>
		<p align="center"><img width="%100" height="30" src="https://cvpr2019.thecvf.com/images/sponsors/cvf_.jpg"> CVPR 2020
			<br>[<b><a href="https://arxiv.org/abs/1912.08193" target="_blank">arXiv</a></b> | <a href="https://www.youtube.com/watch?v=yvNZGDZC3F8" target="_blank"><b>Video</b></a>  | <a href="https://github.com/facebookresearch/detectron2/tree/master/projects" target="_blank"><b>Project</b></a> | <a href="https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend" target="_blank"><b>Code</b></a>]
		</p>
	</td>
</tr>
<tr>
	<td><font size="4">2.</font></td>
	<td><center><a href="https://www.youtube.com/watch?v=sysySMr3YN4" target="_blank"><img src="https://img-blog.csdnimg.cn/img_convert/3dd379d7f61591dc731adfcdf032b91b.png" width="820" height="%100" /></a></center>
	</td>
	<td>
		<p align="center"><font size="5"><b>PolarMask</b>: Single Shot Instance Segmentation with Polar Representation</font>
			<br><b>Enze Xie<sup>∗</sup></b>, Peize Sun, Xiaoge Song, Wenhai Wang, Ding Liang, Chunhua Shen, Ping Luo
		</p>
		<p align="left">
	        <font size =2> <b>Abstract</b>. In this paper, we introduce an anchor-box free and single shot instance segmentation method, which is conceptually simple, fully convolutional and can be used as a mask prediction module for instance segmentation, by easily embedding it into most off-the-shelf detection methods. Our method, termed PolarMask, formulates the instance segmentation problem as instance center classification and dense distance regression in a polar coordinate. Moreover, we propose two effective approaches to deal with sampling high-quality center examples and optimization for dense distance regression, respectively, which can significantly improve the performance and simplify the training process. 
		<details>
  		    <summary>Click to expand</summary>Without any bells and whistles, PolarMask achieves 32.9% in mask mAP with single-model and single-scale training/testing on challenging COCO dataset. For the first time, we demonstrate a much simpler and flexible instance segmentation framework achieving competitive accuracy. We hope that the proposed PolarMask framework can serve as a fundamental and strong baseline for single shot instance segmentation tasks.
		</details>
	        </font>
	     	</p>
		<p align="center"><img width="%100" height="30" src="https://cvpr2019.thecvf.com/images/sponsors/cvf_.jpg"> CVPR 2020
			<br>[<b><a href="https://arxiv.org/abs/1909.13226" target="_blank">arXiv</a></b> | <a href="https://www.youtube.com/watch?v=sysySMr3YN4" target="_blank"><b>Video</b></a> | <a href="https://github.com/xieenze/PolarMask" target="_blank"><b>Code</b></a>  | <a href="https://zhuanlan.zhihu.com/p/84890413" target="_blank"><b>Study</b></a>]
		</p>
	</td>
</tr>
<tr>
	<td><font size="4">3.</font></td>
	<td><center><a href="https://github.com/JizhiziLi/animal-matting" target="_blank"><img src="https://github.com/JizhiziLi/animal-matting/raw/master/demo/src/sample2.jpg" alt="End-to-end Animal Image Matting" width="90" height="%100" /><img src="https://github.com/JizhiziLi/animal-matting/raw/master/demo/src/sample2.png" alt="End-to-end Animal Image Matting" width="90" height="%100" /></a></center>
	</td>
	<td>
		<p align="center"><font size="5"><b>End-to-end Animal Image Matting</b></font>
			<br><b>Jizhizi Li<sup>∗</sup></b>, Jing Zhang, Stephen J. Maybank, Dacheng Tao<sup>∗</sup>
		</p>
		<p align="left">
	        <font size =2> <b>Abstract</b>. Extracting accurate foreground animals from natural animal images benefits many downstream applications such as film production and augmented reality. However, the various appearance and furry characteristics of animals challenge existing matting methods, which usually require extra user inputs such as trimap or scribbles. To resolve these problems, we study the distinct roles of semantics and details for image matting and decompose the task into two parallel sub-tasks: high-level semantic segmentation and low-level details matting. Specifically, we propose a novel Glance and Focus Matting network (GFM), which employs a shared encoder and two separate decoders to learn both tasks in a collaborative manner for end-to-end animal image matting. 
		<details>
  		    <summary>Click to expand</summary>Besides, we establish a novel Animal Matting dataset (AM-2k) containing 2,000 high-resolution natural animal images from 20 categories along with manually labeled alpha mattes. Furthermore, we investigate the domain gap issue between composite images and natural images systematically by conducting comprehensive analyses of various discrepancies between foreground and background images. We find that a carefully designed composition route RSSN that aims to reduce the discrepancies can lead to a better model with remarkable generalization ability. Comprehensive empirical studies on AM-2k demonstrate that GFM outperforms state-of-the-art methods and effectively reduces the generalization error.
		</details>
	        </font>
	     	</p>
		<p align="center"><img width="%100" height="30" src="https://cvpr2019.thecvf.com/images/sponsors/cvf_.jpg"> CVPR 2020
			<br>[<b><a href="https://arxiv.org/abs/2010.16188" target="_blank">arXiv</a></b> | <a href="https://github.com/JizhiziLi/animal-matting" target="_blank"><b>Project Page</b></a> |  <b>Video</b>  |  <a href="https://github.com/JizhiziLi/animal-matting" target="_blank"><b>Code</b></a> | <a href="https://arxiv.org/pdf/2009.06613.pdf" target="_blank"><b>Related Work</b>]
		</p>
	</td>
</tr>
</table>
<br>