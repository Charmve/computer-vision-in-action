## SqueezeNet



![image](https://user-images.githubusercontent.com/29084184/130394576-633dedf0-0435-41f3-9713-da008c5175ee.png)

SqueezeNet系列是比较早期且经典的轻量级网络，SqueezeNet使用Fire模块进行参数压缩，而SqueezeNext则在此基础上加入分离卷积进行改进。虽然SqueezeNet系列不如MobieNet使用广泛，但其架构思想和实验结论还是可以值得借鉴的。


![image](https://user-images.githubusercontent.com/29084184/130395217-ca94e76e-6ab2-426f-ac2a-6e7238f09d04.png)

图1 SqueezeNet 微架构示意图

![image](https://user-images.githubusercontent.com/29084184/130394633-d0c41ef6-7ea3-474c-bfe1-be13c96ee271.png)

图2 

SqueezeNet的核心模块为Fire模块，结构如图1所示，输入层先通过squeeze卷积层($1×1$ 卷积)进行维度压缩，然后通过expand卷积层( $1×1$ 卷积和$3×3$1×1$$ 卷积混合)进行维度扩展。Fire模块包含3个参数，分别为squeeze层的 $1×1$ 卷积核数$s_{1×1}$ 、expand层的 $1×1$ 卷积核数$e_{1×1}$ 和expand层的$1×1$ 卷积核数$e_{3×3}$ ，一般 $s_{1×1} < e_{1×1} + e_{3×3}$ .



- 论文地址：https://arxiv.org/pdf/1602.07360.pdf
- 论文代码：https://github.com/forresti/SqueezeNet