## FBNet

FBNet系列是完全基于NAS搜索的轻量级网络系列，分析当前搜索方法的缺点，逐步增加创新性改进，FBNet结合了DNAS和资源约束，FBNetV2加入了channel和输入分辨率的搜索，FBNetV3则是使用准确率预测来进行快速的网络结构搜索。

![image](https://user-images.githubusercontent.com/29084184/130395378-bf16342d-8e5f-4823-b107-cb1328075661.png)

论文提出FBNet，使用可微神经网络搜索(DNAS)来发现硬件相关的轻量级卷积网络，流程如图1所示。DNAS方法将整体的搜索空间表示为超网，将寻找最优网络结构问题转换为寻找最优的候选block分布，通过梯度下降来训练block的分布，而且可以为网络每层选择不同的block。为了更好地估计网络的时延，预先测量并记录了每个候选block的实际时延，在估算时直接根据网络结构和对应的时延累计即可。

![image](https://user-images.githubusercontent.com/29084184/130395445-38aa6eab-30f0-4bcf-8898-c4a1a852f9b8.png)

- 论文地址：https://arxiv.org/abs/1812.03443
- 论文代码：https://github.com/facebookresearch/mobile-vision