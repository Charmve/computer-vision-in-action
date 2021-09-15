🌍 [English](README.md) | 简体中文| [日本語](README-jp_JP.md) | [Українською](README-uk_UA.md)
<br>

<div align="center">
	<a href="https://charmve.github.io/L0CV-web">
		<img src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/header.svg" width="90%" alt="Click to see the more details">
	</a>
    <br>
    <p>
    <strong>一个简单易用的计算机视觉 MLOps 工具链</strong>
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
<b>注意：如有任何建议、更正和反馈，请提出问题。</b>

这个 repo 的目标是构建一个易于使用的计算机视觉 MLOps 工具链，以实现的基本需要，如模型构建、监控、配置、测试、打包、部署、CI/CD 等。

实现方法参考：https://github.com/graviraja/MLOps-Basics

<br>

----

### 特性 <a name="index"></a>

- 📕 [L0CV 概述](#📘-L0CV概述)
- 🍃 [模型版本控制 - DVC](#-模型版本控制---dvc)
- ⛳ [模型打包 - ONNX](#-模型打包---onnx)
- 🐾 [模型打包 - Docker](#-模型打包---docker)
- 🍀 [CI/CD - GitHub Actions](#-cicd---github-actions)
- ⭐️ [无服务器架构 - AWS Lambda](#-无服务器架构---aws-lambda)
- 🌴 [容器注册表 - AWS ECR](#-容器注册表---aws-ecr)
- ⏳ [无服务器架构 - AWS Lambda](#-无服务器架构---aws-lambda) 

----
<br>

## 📘 L0CV 概述
<p align="center">
  <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/res/ui/l0cv-architecture.png" alt="L0CV architecture" title="L0CV architecture">
</p>

> **版权所有，🈲️ 止商用**
> 本图片采用作者独立研发的图片水印技术作为版权认证 CC-BY-NC 4.0（署名可任意转载），[点击查看](https://github.com/Charmve/computer-vision-in-action/blob/main/res/ui/arch-encryption.png?raw=true)。

<br>

## 🍃 模型版本控制 - DVC
DVC 通常与 Git 一起运行。 Git 照常用于存储和版本代码（包括 DVC 元文件）。 DVC 有助于在 Git 之外无缝存储数据和模型文件，同时保留几乎与存储在 Git 本身相同的用户体验。 为了存储和共享数据缓存，DVC 支持多个远程 - 任何云（S3、Azure、Google Cloud 等）或任何本地网络存储（例如，通过 SSH）。

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/how_dvc_works.gif" alt="DVC" title="DVC">
</p>


DVC 管道（计算图）功能将代码和数据连接在一起。 可以明确指定生成模型所需的所有步骤：输入依赖项，包括数据、要运行的命令和要保存的输出信息。 请参阅下面的快速入门部分或入门教程以了解更多信息。

[Index](#-l0cv-概述)

## ⛳ 模型打包 - ONNX

为什么我们需要模型打包？ 可以使用任何可用的机器学习框架（sklearn、tensorflow、pytorch 等）构建模型。 我们可能想要在不同的环境中部署模型，比如（移动、网络、树莓派）或者想要在不同的框架中运行（在 pytorch 中训练，在 tensorflow 中进行推理）。 使 AI 开发人员能够使用具有各种框架、工具、运行时和编译器的模型的通用文件格式将大有帮助。

这是由社区项目 ONNX 实现的。

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/onnx.jpeg" width="80%" alt="Model Packaging - ONNX" title="Model Packaging - ONNX">
</p>

所用的框架为：
- [ONNX](https://onnx.ai/)
- [ONNXRuntime](https://www.onnxruntime.ai/)

[Index](#-l0cv-概述)

## 🐾 模型打包 - Docker

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>

为什么我们需要打包？ 我们可能不得不与其他人共享我们的应用程序，当他们尝试运行该应用程序时，由于依赖关系问题/操作系统相关问题，大多数时间它无法运行，为此，我们说（工程师之间的名言）它有效 在我的笔记本电脑/系统上。

因此，其他人要运行应用程序，他们必须设置与在主机端运行相同的环境，这意味着需要大量手动配置和安装组件。

这些限制的解决方案是一种称为``容器的技术``。

通过容器化/打包应用程序，我们可以在任何云平台上运行应用程序，以获得托管服务、自动扩展和可靠性等优势。

对应用程序进行打包最突出的工具是 Docker 🛳

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/docker_flow.png" width="60%" alt="Docker" title="Docker">
</p>

[Index](#-l0cv-概述)

## 🍀 CI/CD - GitHub Actions

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

CI/CD 是一种编码哲学和一组实践，您可以使用它来持续构建、测试和部署迭代代码更改。

这种迭代过程有助于减少您基于有缺陷或失败的先前版本开发新代码的机会。 使用这种方法，您可以努力减少从开发新代码到部署的人工干预，甚至根本不需要干预。

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/basic_flow.png" width="80%" alt="CI/CD" title="CI/CD">
</p>

[Index](#-l0cv-概述)

## 🌴 容器注册表 - AWS ECR

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

容器注册表是存储容器镜像的地方。 容器镜像是一个由多个层组成的文件，可以在单个实例中执行应用程序。 将所有图像托管在一个存储位置允许用户在需要时提交、识别和拉取图像。

Amazon Simple Storage Service (S3) 是一种用于 Internet 的存储。 它专为跨多个地理区域提供大容量、低成本的存储而设计。

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/ecr_flow.png" width="80%" alt="AWS ECR" title="AWS ECR">
</p>

## ⭐️ 无服务器架构 - AWS Lambda

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

无服务器架构是一种无需管理基础架构即可构建和运行应用程序和服务的方法。 该应用程序仍然在服务器上运行，但所有服务器管理都由第三方服务 (AWS) 完成。 我们不再需要配置、扩展和维护服务器来运行应用程序。 通过使用无服务器架构，开发人员可以专注于他们的核心产品，而不必担心在云中或本地管理和操作服务器或运行时。

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/lambda_flow.png" width="80%" alt="AWS Lambda" title="AWS Lambda">
</p>

[Index](#-l0cv-概述)

## ⏳ 预测监视 - Kibana

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

监控系统可以帮助我们相信我们的系统运行顺利，并且在系统出现故障时，可以在诊断根本原因时快速提供适当的上下文。

我们想要在训练和推理过程中监控的东西是不同的。 在训练过程中，我们关心的是损失是否在减少，模型是否过度拟合等。

但是，在推理过程中，我们希望我们的模型能够做出正确的预测。 

<p align="center">
    <img  src="https://github.com/Charmve/computer-vision-in-action/blob/main/L0CV/images/kibana_flow.png" width="80%" alt="Kibana" title="Kibana">
</p>

[Index](#-l0cv-概述)

<br>

[<button>💖 Sponsor me </button>](https://charmve.github.io/L0CV-web/Sponsors.html)

Code with ❤️ & ☕️  [@Charmve](https://github.com/Charmve)
