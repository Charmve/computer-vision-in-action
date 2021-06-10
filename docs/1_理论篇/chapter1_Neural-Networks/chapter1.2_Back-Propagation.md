<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

# 第 1 章 神经网络

作者: 张伟 (Charmve)

日期: 2021/06/09

- 第 1 章 [神经网络](../)
    - 1.1 [Softmax 回归](chapter1.1 Softmax回归.md)
    - 1.2 反向传播算法
      - 1.2.1 [概述](#121-概述)
      - 1.2.2 [前向传播](#122-前向传播)
      - 1.2.3 [反向传播](#123-反向传播)
      - 1.2.4 [Python源代码](#124-Python源代码)
    - 1.3 [多层感知器](chapter1_3_多层感知器MLP.md)
    - 1.4 [神经学观点](chapter1.4_神经学观点.md)
    - 1.5 [实战项目 1 - 手写字分类](https://blog.csdn.net/Charmve/article/details/108531735)
    - 小结
    - 参考文献


## 1.2 反向传播算法

### 1.2.1 概述

反向传播法其实是神经网络的基础了，但是很多人在学的时候总是会遇到一些问题，或者看到大篇的公式觉得好像很难就退缩了，其实不难，就是一个链式求导法则反复用。如果不想看公式，可以直接把数值带进去，实际的计算一下，体会一下这个过程之后再来推导公式，这样就会觉得很容易了。

说到神经网络，大家看到这个图应该不陌生（如图1.1所示）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232127308.png)

图1.1 神经网络结构

这是典型的三层神经网络的基本构成，Layer L1是输入层(input layer)，Layer L2是隐含层(hidden layer)，Layer L3是输出层(output layer)。

我们现在手里有一堆数据{x1,x2,x3,...,xn},输出也是一堆数据{y1,y2,y3,...,yn},现在要它们在隐含层做某种变换，让你把数据灌进去后得到你期望的输出。

如果你希望你的输出和原始输入一样，那么就是最常见的[自编码模型（Auto-Encoder）](../../../3_进阶篇/chapter12-生成对抗模型/chapter12_2-自编码器Auto-encoder.md)。

**可能有人会问，为什么要输入输出都一样呢？有什么用啊？**

其实应用挺广的，在图像识别，文本分类等等都会用到，我会专门再写一篇Auto-Encoder的文章来说明，包括一些变种之类的。如果你的输出和原始输入不一样，那么就是很常见的人工神经网络了，相当于让原始数据通过一个映射来得到我们想要的输出数据，也就是我们今天要讲的话题。

本文直接举一个例子，带入数值演示反向传播法的过程，公式的推导等到下次写Auto-Encoder的时候再写，其实也很简单，感兴趣的同学可以自己推导下试试。

假设，你有这样一个网络层，如图1.2所示。

![image](https://user-images.githubusercontent.com/29084184/121332556-8faa5600-c94a-11eb-817a-fb7b74becde2.png)

图1.2 神经网络的网络层 [4]

第一层是输入层，包含两个神经元i1，i2，和截距项b1；第二层是隐含层，包含两个神经元h1,h2和截距项b2，第三层是输出o1,o2，每条线上标的wi是层与层之间连接的权重，激活函数我们默认为 [sigmoid函数](http://mp.weixin.qq.com/s?__biz=MzIxMjg1Njc3Mw==&mid=2247484495&idx=1&sn=0bbb2094d93169baf20eedb284bc668f&chksm=97befee5a0c977f332a4381ffc9b4285c94acc544317dd39c44a0c2ec58c86b7b0e3a5933d9c&scene=21#wechat_redirect)，更多激活函数详解可在本书附件中查看。

现在对他们赋上初值，如下图1.3所示。

![image](https://user-images.githubusercontent.com/29084184/121332619-9e910880-c94a-11eb-822e-c666718d0cb7.png)

图1.3 网络层增加权重

```
其中，输入数据  i1=0.05，i2=0.10;　　　
      输出数据  o1=0.01,o2=0.99;　　　
      初始权重  w1=0.15,w2=0.20,w3=0.25,w4=0.30;　　　　　　　  
               w5=0.40,w6=0.45,w7=0.50,w8=0.55
```

**目标：给出输入数据 i1,i2 (0.05和0.10)，使输出尽可能与原始输出 o1,o2 (0.01和0.99)接近。**

### 1.2.2 前向传播
#### 1.2.2.1 输入层---->隐含层：

计算神经元h1的输入加权和：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232554501.png#pic_center)
           

神经元h1的输出o1:(此处用到激活函数为sigmoid函数)：

 ![在这里插入图片描述](https://img-blog.csdnimg.cn/2020052923260350.png#pic_center)        

同理，可计算出神经元h2的输出o2：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232614652.png#pic_center)
                                  

#### 1.2.2.2 隐含层---->输出层：

计算输出层神经元o1和o2的值：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232622747.png)


这样前向传播的过程就结束了，我们得到输出值为[0.75136079 , 0.772928465]，与实际值[0.01 , 0.99]相差还很远，现在我们对误差进行反向传播，更新权值，重新计算输出。

### 1.2.3 反向传播
#### 1.2.3.1 计算总误差

总误差：(square error)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232635706.png#pic_center)

但是有两个输出，所以分别计算o1和o2的误差，总误差为两者之和：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232640654.png)

#### 1.2.3.2 隐含层---->输出层的权值更新：

以权重参数w5为例，如果我们想知道w5对整体误差产生了多少影响，可以用整体误差对w5求偏导求出：（链式法则）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232645863.png#pic_center)


如图1.4 所示，可以更直观的看清楚误差是怎样反向传播的：

![image](https://user-images.githubusercontent.com/29084184/121332670-a81a7080-c94a-11eb-9264-e695bedde68e.png)

图1.4 误差的反向传播

现在我们来分别计算每个式子的值：

计算 $\frac{\partial E_{total}}{\partial out_{o1}}$  ：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232659709.png)


计算 $\frac{\partial out_{o1}}{\partial net_{o1}}$  ：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232842179.png)

（这一步实际上就是对sigmoid函数求导，比较简单，可以自己推导一下）

计算 $\frac{\partial net_{o1}}{\partial w_{5}}$  ：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232850387.png)


最后三者相乘：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232856845.png)

这样我们就计算出整体误差$E(total)$对$w5$的偏导值。

回过头来再看看上面的公式，我们发现：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232902266.png)

为了表达方便，用 $\delta_{o1}$ 来表示输出层的误差：

![在这里插入图片描述](https://img-blog.csdnimg.cn/202005292329076.png)

因此，整体误差E(total)对w5的偏导公式可以写成：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232930326.png)

如果输出层误差计为负的话，也可以写成：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232938102.png)


最后我们来更新$w5$的值：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529232943758.png)

（其中，$\eta$  是学习速率，这里我们取0.5）

同理，可更新w6,w7,w8:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529233004225.png)

#### 1.2.3.3 隐含层---->隐含层的权值更新：

方法其实与上面说的差不多，但是有个地方需要变一下，在上文计算总误差对w5的偏导时，是从out(o1)---->net(o1)---->w5,但是在隐含层之间的权值更新时，是out(h1)---->net(h1)---->w1,而out(h1)会接受E(o1)和E(o2)两个地方传来的误差，所以这个地方两个都要计算，如图1.5所示。

![image](https://user-images.githubusercontent.com/29084184/121332725-b6688c80-c94a-11eb-8c25-d167475f3dfc.png)

图1.5 隐含层---->隐含层的权值更新

计算 $\frac{\partial E_{total}}{\partial out_{h1}}$  ：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529233020502.png#pic_center)

先计算 $\frac{\partial E_{o1}}{\partial out_{h1}}$  ：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529233025870.png)

同理，计算出：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529233029879.png#pic_center)　　　　　　　　　

两者相加得到总值：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529233044548.png#pic_center)


再计算 $\frac{\partial out_{h1}}{\partial net_{h1}}$  ：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020052923303965.png#pic_center)


再计算 $\frac{\partial net_{h1}}{\partial w_{1}}$ ：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529233051731.png#pic_center)


最后，三者相乘：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529233056940.png#pic_center)


为了简化公式，用$sigma(h1)$表示隐含层单元h1的误差：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529233103731.png)

最后，更新$w1$的权值：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529233108181.png#pic_center)

同理，额可更新$w2,w3,w4$的权值：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529233112721.png)

这样误差反向传播法就完成了，最后我们再把更新的权值重新计算，不停地迭代，在这个例子中第一次迭代之后，总误差E(total)由0.298371109下降至0.291027924。迭代10000次后，总误差为0.000035085，输出为[0.015912196, 0.984065734]（原输入为[0.01, 0.99]）,证明效果还是不错的。

### 1.2.4 Python源代码

```python
#coding:utf-8
import random
import math

#
#   参数解释：
#   "pd_" ：偏导的前缀
#   "d_" ：导数的前缀
#   "w_ho" ：隐含层到输出层的权重系数索引
#   "w_ih" ：输入层到隐含层的权重系数的索引


class NeuralNetwork:
    LEARNING_RATE = 0.5


    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs


        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)


        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)


    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1


    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1


    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')


    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)


    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)


        # 1. 输出神经元的值
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):


            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])


        # 2. 隐含层神经元的值
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):


            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]


            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()


        # 3. 更新输出层权重系数
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):


                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)


                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight


        # 4. 更新隐含层的权重系数
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):


                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)


                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight


    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error


class NeuronLayer:
    def __init__(self, num_neurons, bias):


        # 同一层的神经元共享一个截距项b
        self.bias = bias if bias else random.random()


        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))


    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)


    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs


    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []


    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output


    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias


    # 激活函数sigmoid
    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();


    # 每一个神经元的误差是由平方差公式计算的
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

# 文中的例子:
nn = NeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
for i in range(10000):
    nn.train([0.05, 0.1], [0.01, 0.09])
    print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.09]]]), 9))

#另外一个例子，可以把上面的例子注释掉再运行一下:

# training_sets = [
#     [[0, 0], [0]],
#     [[0, 1], [1]],
#     [[1, 0], [1]],
#     [[1, 1], [0]]
# ]


# nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
# for i in range(10000):
#     training_inputs, training_outputs = random.choice(training_sets)
#     nn.train(training_inputs, training_outputs)
#     print(i, nn.calculate_total_error(training_sets))
```


稳重使用的是sigmoid激活函数，实际还有几种不同的激活函数可以选择，具体的可以参考文献[3]，最后推荐一个在线演示神经网络变化的网址：http://www.emergentmind.com/neural-network ，可以自己填输入输出，然后观看每一次迭代权值的变化，很好玩。


### 小结


### 参考文献

1. Poll的笔记：[Mechine Learning & Algorithm]神经网络基础. http://www.cnblogs.com/maybe2030/p/5597716.html#3457159

2. Rachel_Zhang. Stanford机器学习---第五讲. 神经网络的学习 Neural Networks learning. http://blog.csdn.net/abcjennifer/article/details/7758797

3. Sargur Srihari. Backpropagation. http://www.cedar.buffalo.edu/%7Esrihari/CSE574/Chap5/Chap5.3-BackProp.pdf

4. Matt Mazur. A Step by Step Backpropagation Example. https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

