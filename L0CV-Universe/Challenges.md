<p align="center">
  <img src="../res/ui/challenges.png" title="L0CV Challenges">
</p>

``L0CV-Challenges`` 是面向各计算机视觉任务的 Baseline 复现及提高，组织一起志同道合的小伙伴一起复现最新论文。


## Challenges

<table class="table table-striped table-bordered table-vcenter">
    <tbody class=ai-notebooks-table-content>
    <tr>
        <td colspan="1" rowspan="1" class="ai-notebooks-table-points ai-orange-link">
        <div class="mdl-cell mdl-cell--5-col mdl-cell--middle">
            <div class="content">
            <h3 style="text-align:center">1️⃣ 视频补全</h3>
            <p><b>[ECCV 2020] Flow-edge Guided Video Completion</b></p>
            </div>
            <div class="mdl-grid running" align="center" style="text-align:center">
              <div class="running-item">
                    <a href="https://arxiv.org/abs/2009.01835">
                    <img src="https://user-images.githubusercontent.com/29084184/131961075-eb57927b-bbcc-4c0f-b361-87c326be9dcc.png" width="50">
                    <p>Paper</p>
                    </a>
                </div>
                <div class="running-item">
                    <a href="http://chengao.vision/FGVC/">
                    <img src="https://github.com/Charmve/computer-vision-in-action/blob/main/res/ui/laptop_jupyter.png" width="50">
                    <p>Project Website</p>
                    </a>
                </div>
                <div class="running-item">
                    <a href="https://colab.research.google.com/drive/1pb6FjWdwq_q445rG2NP0dubw7LKNUkqc?usp=sharing">
                    <img src="https://github.com/Charmve/computer-vision-in-action/blob/main/res/ui/colab.png" width="50">
                    <p>Google<br>Colab</p>
                    </a>
                </div>
            </div>
        </div>
        </td>
        <td>
            <div class="mdl-cell mdl-cell--7-col">
                <p align='center'>
                  <img src='http://chengao.vision/FGVC/files/FGVC_teaser.png' width='900'/>
                </p>
                主要任务是对视频缺失部分进行补全，在去水印、logo、马赛克上有广泛应用，常见算 法可以参考Deepfill，该算法基本流程是计算稠密光流（RAFT）、计算边缘（Canny）、补全边缘（EdgeConnect）、补全光流、传播RGB值。
            </div>
        </td>
    </tr>
    </tbody>
</table>

<table class="table table-striped table-bordered table-vcenter">
    <tbody class=ai-notebooks-table-content>
    <tr>
        <td colspan="1" rowspan="1" class="ai-notebooks-table-points ai-orange-link">
        <div class="mdl-cell mdl-cell--5-col mdl-cell--middle">
            <div class="content">
            <h3 style="text-align:center">2️⃣ 虚拟换装</h3>
            <p><b>VOGUE: Try-On by StyleGAN Interpolation Optimization</b></p>
            </div>
            <div class="mdl-grid running" align="center" style="text-align:center">
              <div class="running-item">
                    <a href="http://arxiv.org/abs/2101.02285">
                    <img src="https://user-images.githubusercontent.com/29084184/131961075-eb57927b-bbcc-4c0f-b361-87c326be9dcc.png" width="50">
                    <p>Paper</p>
                    </a>
                </div>
                <div class="running-item">
                    <a href="https://github.com/Charmve/VOGUE-Try-On/blob/main/web_home/demo_rewrite.html">
                    <img src="https://github.com/Charmve/computer-vision-in-action/blob/main/res/ui/laptop_jupyter.png" width="50">
                    <p>Project Website</p>
                    </a>
                </div>
                <div class="running-item">
                    <a href="https://github.com/Charmve/VOGUE-Try-On">
                    <img src="https://user-images.githubusercontent.com/29084184/131961742-30eb5b87-df80-48c7-b193-8231e3adb423.png" width="50">
                    <p>GitHub<br>Code</p>
                    </a>
                </div>
            </div>
        </div>
        </td>
        <td>
            <div class="mdl-cell mdl-cell--7-col">
                <p align='center'>
                  <img src='https://github.com/Charmve/VOGUE-Try-On/raw/main/ui/VOGUE.png' width='900'/>
                </p>
                主要任务是对给定人物图片换上另一服饰，降低混淆区域。
            </div>
        </td>
    </tr>
    </tbody>
</table>

<table class="table table-striped table-bordered table-vcenter">
    <tbody class=ai-notebooks-table-content>
    <tr>
        <td colspan="5" rowspan="1" class="ai-notebooks-table-points ai-orange-link">
            <div class="content">
              <h3 style="text-align:center">3️⃣ 背景抠图</h3>
              <p align='center'>
                  <img src='https://github.com/PeterL1n/Matting-PyTorch/raw/master/images/teaser.gif?raw=true' width='900'/>
                </p>
              <p><b>Real-Time High-Resolution Background Matting</b></p>
            </div>
          </td>
      </tr>
        <div class="mdl-cell mdl-cell--5-col mdl-cell--middle">
            <div class="mdl-grid running" align="center" style="text-align:center">
                <div class="running-item">
                  <tr>
                  <td>
                    <a href="https://arxiv.org/abs/2012.07810">
                    <img src="https://user-images.githubusercontent.com/29084184/131961075-eb57927b-bbcc-4c0f-b361-87c326be9dcc.png" width="50">
                    <br>Paper
                    </a>
                  </td>
                  <td>
                    <a href="https://grail.cs.washington.edu/projects/background-matting-v2/#/">
                    <img src="https://github.com/Charmve/computer-vision-in-action/blob/main/res/ui/laptop_jupyter.png" width="50">
                    <br>Project Website
                    </a>
                  </td>
                  <td>
                    <a href="https://colab.research.google.com/drive/1cTxFq1YuoJ5QPqaTcnskwlHDolnjBkB9?usp=sharing">
                    <img src="https://github.com/Charmve/computer-vision-in-action/blob/main/res/ui/colab.png" width="50">
                    <br>Google Colab
                    </a>
                  </td>
                  <td>
                    <a href="https://github.com/PeterL1n/BackgroundMattingV2">
                    <img src="https://user-images.githubusercontent.com/29084184/131961742-30eb5b87-df80-48c7-b193-8231e3adb423.png" width="50">
                    <br>GitHub
                    </a>
                  </td>
                  <td>
                    <a href="https://mp.weixin.qq.com/s/MRjiZhv9ysWwF-pLrS0fjQ">
                    <img src="https://github.com/Charmve/TimeWarp/raw/main/images/ui/logo_V.png" width="50">
                    <br>Evolution Work
                    </a>
                  </td>
                </div>
            </div>
          </div>
      </tr>
      <tr>
        <td colspan="5" rowspan="1" class="ai-notebooks-table-points ai-orange-link">
            <div class="mdl-cell mdl-cell--7-col">
                主要任务是对复杂背景进行抠图，应用于在线会议、直播等场景。
            </div>
        </td>
    </tr>
    </tbody>
</table>

<table class="table table-striped table-bordered table-vcenter">
    <tbody class=ai-notebooks-table-content>
    <tr>
        <td colspan="1" rowspan="1" class="ai-notebooks-table-points ai-orange-link">
        <div class="mdl-cell mdl-cell--5-col mdl-cell--middle">
            <div class="content">
            <h3 style="text-align:center">4️⃣ 可通行区域检测</h3>
            <p><b>SNE-RoadSeg: Incorporating Surface Normal Information into Semantic Segmentation for Accurate Freespace Detection</b></p>
            </div>
            <div class="mdl-grid running" align="center" style="text-align:center">
              <div class="running-item">
                    <a href="http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750341.pdf">
                    <img src="https://user-images.githubusercontent.com/29084184/131961075-eb57927b-bbcc-4c0f-b361-87c326be9dcc.png" width="50">
                    <p>Paper</p>
                    </a>
                </div>
                <div class="running-item">
                    <a href="https://github.com/Charmve/SNE-RoadSeg2">
                    <img src="https://github.com/Charmve/computer-vision-in-action/blob/main/res/ui/laptop_jupyter.png" width="50">
                    <p>Project Website</p>
                    </a>
                </div>
                <div class="running-item">
                    <a href="https://github.com/Charmve/SNE-RoadSeg2">
                    <img src="https://user-images.githubusercontent.com/29084184/131961742-30eb5b87-df80-48c7-b193-8231e3adb423.png" width="50">
                    <p>GitHub<br>Code</p>
                    </a>
                </div>
            </div>
        </div>
        </td>
        <td>
            <div class="mdl-cell mdl-cell--7-col">
                <p align='center'>
                  <img src='https://github.com/Charmve/SNE-RoadSeg2/raw/master/doc/kitti.gif' width='900'/>
                </p>
                主要任务是对道路可通行区域做检测，运用最新分割网络``SNE-RoadSeg``。
            </div>
        </td>
    </tr>
    </tbody>
</table>
  
## Join Us

加微信预约报名，组成战队，自主学习！ VX: Yida_Zhang2 (备注：Challenges-昵称-项目序号)

🔥 Keep Going ...
