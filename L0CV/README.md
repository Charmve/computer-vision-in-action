🌍 English | [简体中文](README-zh_CN.md)| [日本語](README-jp_JP.md) | [Українською](README-uk_UA.md)
<br>

<div align="center">
	<a href="https://charmve.github.io/L0CV-web">
		<img src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/header.svg" width="90%" alt="Click to see the more details">
	</a>
    <br>
    <p>
    Easy-to-go tool-chain for computer vision with <a href="">MLOps</a>, <a href="">AutoML</a> and <a href="">Data Security</a>
    </p>
    <p align="center">
        <a href="https://circleci.com/gh/Charmve/computer-vision-in-action"><img src="https://circleci.com/gh/Charmve/computer-vision-in-action.svg?style=svg" alt="CircleCI" title="CircleCI"></a>
        <a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank" style="display:inline-block"><img src="https://img.shields.io/badge/license-Apache%202.0-red?logo=apache" alt="Code License"></a>
        <a href="https://mybinder.org/v2/gh/Charmve/computer-vision-in-action/main/notebooks/"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>
        <a href="https://github.com/Charmve/computer-vision-in-action/tree/main/code/"><img src="https://img.shields.io/badge/Run%20on-Colab-000000.svg?logo=googlecolab&color=yellow" alt="Run on Colab"></a>
    </p>
    <p align="center">
        <a href="https://github.com/Charmve/computer-vision-in-action/tree/main/code">Quickstart</a> •
        <a href="https://github.com/Charmve/computer-vision-in-action/tree/main/notebooks">Notebook</a> •
        <a href="https://github.com/Charmve/computer-vision-in-action/issues">Community</a>  •
        <a href="https://charmve.github.io/computer-vision-in-action/">Docs</a> 
    </p>
</div>


----



### Features <a name="index"></a>

- 📕 [Summary](#📘-summary)
- 🍃 [Data & Models Version Control - DVC](#🍃-DVC-)
- ⛳ [Model Packaging - ONNX](#⛳-ONNX-)
- 🐾 [Model Packaging - Docker](#🐾-Docker-)
- 🍀 [CI/CD - GitHub Actions](#🍀-CICD-)
- ⭐️ [Serverless Deployment - AWS Lambda](#⭐️-lambda-)
- 🌴 [Container Registry - AWS ECR](#🌴-AWSECR-)
- ⏳ [Prediction Monitoring - Kibana](#⏳-kibana-) 


## 📘 Summary<a name="summary"></a>

<div>
    <p align="center"><img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/res/ui/l0cv-architecture.png" alt="L0CV architecture" title="L0CV architecture"></p>
    <p align="right">
        <details><summary><b>版权所有，🈲️ 止商用</b></summary>
        * This image uses independently developed image watermarking technology for copyright certification, <a href="https://github.com/Charmve/computer-vision-in-action/blob/main/res/ui/arch-encryption.png?raw=true" target="_blank"><b>as shown in the figure</b></a>.
        <br>本图片采用作者独立研发的图片水印技术作为版权认证 <kbd>CC-BY-NC 4.0</kbd>（署名可任意转载），<a href="https://github.com/Charmve/computer-vision-in-action/blob/main/res/ui/arch-encryption.png?raw=true" target="_blank"><b>点击查看</b></a>。
        </details>
    </p>
</div>

<br>

# L0CV-MLOps

<b>Note: Please raise an issue for any suggestions, corrections, and feedback.</b>

The goal of this repo is to build an easy-to-go computer vision tool-chain to realise the basics of MLOps like model building, monitoring, configurations, testing, packaging, deployment, cicd, etc.

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/summary.png" alt="MLOps" title="MLOps">
</p>


## 🍃 Data & Models Version Control - DVC <a name="DVC"></a>

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>

DVC usually runs along with Git. Git is used as usual to store and version code (including DVC meta-files). DVC helps to store data and model files seamlessly out of Git, while preserving almost the same user experience as if they were stored in Git itself. To store and share the data cache, DVC supports multiple remotes - any cloud (S3, Azure, Google Cloud, etc) or any on-premise network storage (via SSH, for example).

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/how_dvc_works.gif" alt="DVC" title="DVC">
</p>

The DVC pipelines (computational graph) feature connects code and data together. It is possible to explicitly specify all steps required to produce a model: input dependencies including data, commands to run, and output information to be saved. See the quick start section below or the Get Started tutorial to learn more.


- reference - https://github.com/iterative/dvc

[Index](#features)

## ⛳ Model Packaging - ONNX <a name="ONNX"></a>

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

Why do we need model packaging? Models can be built using any machine learning framework available out there (sklearn, tensorflow, pytorch, etc.). We might want to deploy models in different environments like (mobile, web, raspberry pi) or want to run in a different framework (trained in pytorch, inference in tensorflow). A common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers will help a lot.

This is acheived by a community project ``ONNX``.

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/onnx.jpeg" width="80%" alt="Model Packaging - ONNX" title="Model Packaging - ONNX">
</p>

Following tech stack is used:

- [ONNX](https://onnx.ai/)
- [ONNXRuntime](https://www.onnxruntime.ai/)


[Index](#features)


## 🐾 Model Packaging - Docker <a name="Docker"></a>

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>

Why do we need packaging? We might have to share our application with others, and when they try to run the application most of the time it doesn’t run due to dependencies issues / OS related issues and for that, we say (famous quote across engineers) that It works on my laptop/system.

So for others to run the applications they have to set up the same environment as it was run on the host side which means a lot of manual configuration and installation of components.

The solution to these limitations is a technology called **Containers**.

By containerizing/packaging the application, we can run the application on any cloud platform to get advantages of managed services and autoscaling and reliability, and many more.

The most prominent tool to do the packaging of application is Docker 🛳

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/docker_flow.png" width="60%" alt="Docker" title="Docker">
</p>

[Index](#features)


## 🍀 CI/CD - GitHub Actions <a name="CICD"></a>

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

Refer to the [Blog Post here](https://www.ravirajag.dev/blog/mlops-github-actions)

CI/CD is a coding philosophy and set of practices with which you can continuously build, test, and deploy iterative code changes.

This iterative process helps reduce the chance that you develop new code based on a buggy or failed previous versions. With this method, you strive to have less human intervention or even no intervention at all, from the development of new code until its deployment.

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/basic_flow.png" width="80%" alt="CI/CD" title="CI/CD">
</p>

[Index](#features)

## 🌴 Container Registry - AWS ECR <a name="AWS ECR"></a>

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

A container registry is a place to store container images. A container image is a file comprised of multiple layers which can execute applications in a single instance. Hosting all the images in one stored location allows users to commit, identify and pull images when needed.

Amazon Simple Storage Service (S3) is a storage for the internet. It is designed for large-capacity, low-cost storage provision across multiple geographical regions.

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/ecr_flow.png" width="80%" alt="AWS ECR" title="AWS ECR">
</p>


[Index](#features)


## ⭐️ Serverless Deployment - AWS Lambda <a name="AWS Lambda"></a>

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

A serverless architecture is a way to build and run applications and services without having to manage infrastructure. The application still runs on servers, but all the server management is done by third party service (AWS). We no longer have to provision, scale, and maintain servers to run the applications. By using a serverless architecture, developers can focus on their core product instead of worrying about managing and operating servers or runtimes, either in the cloud or on-premises.

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/lambda_flow.png" width="80%" alt="AWS Lambda" title="AWS Lambda">
</p>


[Index](#features)


## ⏳ Prediction Monitoring - Kibana <a name="Kibana"></a>

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

Monitoring systems can help give us confidence that our systems are running smoothly and, in the event of a system failure, can quickly provide appropriate context when diagnosing the root cause.

Things we want to monitor during and training and inference are different. During training we are concered about whether the loss is decreasing or not, whether the model is overfitting, etc.

But, during inference, We like to have confidence that our model is making correct predictions.

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/kibana_flow.png" width="80%" alt="Kibana" title="Kibana">
</p>




[Index](#features)

<br>

[<button>💖 Sponsor me </button>](https://charmve.github.io/L0CV-web/Sponsors.html)

Code with ❤️ & ☕️




