# Papers on Ternary and Binary Networks

Papers and codes about Ternary and Binary Networks for easier survey and reference. We care about **the accuracy and  the implementation** on hardware for real-world speedup benefits.

## Table of Contents

- [Survey_Papers](#Survey_Papers)  
- [Papers](#Papers)  
	- [BNN](#BNN)  
	- [TNN](#TNN)  
	- [Mixed-Precision](#Mixed-Precision)  
	- [INT8](#INT8)  
	- [Implementation and Acceleration](#ImplementationAndAcceleration)  




## Survey_Papers

[[NeuCom 2021](https://arxiv.org/abs/2011.14808)] Bringing AI To Edge: From Deep Learning’s Perspective

<details><summary>Bibtex</summary><pre><code>@article{AI_Neuro_2021,
title = {Bringing AI To Edge: From Deep Learning’s Perspective},
journal = {Neurocomputing},
year = {2021},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2021.04.141},
url = {https://www.sciencedirect.com/science/article/pii/S0925231221016428},
author = {Di Liu and Hao Kong and Xiangzhong Luo and Weichen Liu and Ravi Subramaniam}
}</code></pre></details>

[[NeuCom 2021](https://arxiv.org/abs/2101.09671v3)] Pruning and Quantization for Deep Neural Network Acceleration: A Survey

<details><summary>Bibtex</summary><pre><code>@article{PandQ_Neuro_2021,
  title={Pruning and quantization for deep neural network acceleration: A survey},
  author={Liang, Tailin and Glossner, John and Wang, Lei and Shi, Shaobo and Zhang, Xiaotong},
  journal={Neurocomputing},
  volume={461},
  pages={370--403},
  year={2021},
  publisher={Elsevier}
}</code></pre></details>

[[PR 2020](https://arxiv.org/abs/2004.03333)] [[Blog](https://mp.weixin.qq.com/s/QGva6fow9tad_daZ_G2p0Q)] Binary Neural Networks: A Survey 

<details><summary>Bibtex</summary><pre><code>@article{Qin:pr20_bnn_survey,
	title = "Binary neural networks: A survey",
	author = "Haotong Qin and Ruihao Gong and Xianglong Liu and Xiao Bai and Jingkuan Song and Nicu Sebe",
	journal = "Pattern Recognition",
	volume = "105",
	pages = "107281",
	year = "2020"
}</code></pre></details>

[[ArXiv 2021](https://arxiv.org/abs/2103.13630)] A Survey of Quantization Methods for Efficient Neural Network Inference

<details><summary>Bibtex</summary><pre><code>@misc{gholami2021survey,
      title={A Survey of Quantization Methods for Efficient Neural Network Inference}, 
      author={Amir Gholami and Sehoon Kim and Zhen Dong and Zhewei Yao and Michael W. Mahoney and Kurt Keutzer},
      year={2021},
      eprint={2103.13630},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}</code></pre></details>

[[ArXiv 2018](https://arxiv.org/abs/1808.04752)] A survey on methods and theories of quantized neural networks

<details><summary>Bibtex</summary><pre><code>@article{Qsuvery_ArXiv_2018,
  title={A survey on methods and theories of quantized neural networks},
  author={Guo, Yunhui},
  journal={arXiv preprint arXiv:1808.04752},
  year={2018}
}</code></pre></details>



## Papers

**Keywords**: **`BNN`**: Binary Neural Networks | **`TBN`** : Ternary-activation Binary-weight Networks| **`TNN`**: Ternary Neural Networks | **`mixed`**: Mixed Precision | **`INT4`**: 4-bit integer quantization | __`INT8`__: 8-bit integer quantization 

**Platforms**: **`CPU`** | **`GPU`** | **`FPGA`** | **`ASIC`** | **`IMC`**: In-Memory-Computing

------

### BNN

[[CVPR_2021](https://openaccess.thecvf.com/content/CVPR2021W/BiVision/html/Razani_Adaptive_Binary-Ternary_Quantization_CVPRW_2021_paper.html)] [__`BWN+TWN`__] Adaptive Binary-Ternary Quantization

<details><summary>Bibtex</summary><pre><code>@InProceedings{BTQ_CVPR_2021,
    author    = {Razani, Ryan and Morin, Gregoire and Sari, Eyyub and Nia, Vahid Partovi},
    title     = {Adaptive Binary-Ternary Quantization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {4613-4618}
}</code></pre></details>

[[ECCV_2020](https://arxiv.org/abs/2003.03488)] [__`BNN, QNN`__] ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions [[PyTorch](https://github.com/liuzechun/ReActNet)]

<details><summary>Bibtex</summary><pre><code>@inproceedings{liu2020reactnet,
  title={ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions},
  author={Liu, Zechun and Shen, Zhiqiang and Savvides, Marios and Cheng, Kwang-Ting},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}</code></pre></details>

[[CVPR_2020](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Pouransari_Least_Squares_Binary_Quantization_of_Neural_Networks_CVPRW_2020_paper.html)] [__`BNN, QNN`__] Least Squares Binary Quantization of Neural Networks [[PyTorch](https://github.com/apple/ml-quant)]

<details><summary>Bibtex</summary><pre><code>@inproceedings{LS-BQNN_CVPR_2020,
  title={Least squares binary quantization of neural networks},
  author={Pouransari, Hadi and Tu, Zhucheng and Tuzel, Oncel},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={698--699},
  year={2020}
}</code></pre></details>

[[CVPR_2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Qin_Forward_and_Backward_Information_Retention_for_Accurate_Binary_Neural_Networks_CVPR_2020_paper.html)] [__`BNN`__] IR-Net: Forward and Backward Information Retention for Accurate Binary Neural Networks [[PyTorch](https://github.com/htqin/IR-Net)]

<details><summary>Bibtex</summary><pre><code>@inproceedings{IR-Net_CVPR_2020,
  title={Forward and backward information retention for accurate binary neural networks},
  author={Qin, Haotong and Gong, Ruihao and Liu, Xianglong and Shen, Mingzhu and Wei, Ziran and Yu, Fengwei and Song, Jingkuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2250--2259},
  year={2020}
}</code></pre></details>

[[ICASSP_2020](https://arxiv.org/abs/1909.12117)] [__`BNN`__] Balanced Binary Neural Networks with Gated Residual

<details><summary>Bibtex</summary><pre><code>@inproceedings{BBG_ICASSP_2020,
  title={Balanced binary neural networks with gated residual},
  author={Shen, Mingzhu and Liu, Xianglong and Gong, Ruihao and Han, Kai},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4197--4201},
  year={2020},
  organization={IEEE}
}</code></pre></details>

[[CVPR_2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Learning_Channel-Wise_Interactions_for_Binary_Convolutional_Neural_Networks_CVPR_2019_paper.html)] [__`BNN`__] CI-BCNN: Learning Channel-Wise Interactions for Binary Convolutional Neural Networks

<details><summary>Bibtex</summary><pre><code>@inproceedings{CI-BCNN_CVPR_2019,
  title={Learning channel-wise interactions for binary convolutional neural networks},
  author={Wang, Ziwei and Lu, Jiwen and Tao, Chenxin and Zhou, Jie and Tian, Qi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={568--577},
  year={2019}
}</code></pre></details>

[[BMVC_2019](https://arxiv.org/abs/1909.13863)] [__`BNN`__] XNOR-Net++: Improved Binary Neural Networks

<details><summary>Bibtex</summary><pre><code>@article{XNOR-NetPlus_BMVC_2019,
  title={{XNOR-Net++}: Improved binary neural networks},
  author={Bulat, Adrian and Tzimiropoulos, Georgios and Center, Samsung AI},
  journal = {The British Machine Vision Conference (BMVC)},
  year = {2019}
}</code></pre></details>

[[ECCV_2018](https://openaccess.thecvf.com/content_ECCV_2018/html/zechun_liu_Bi-Real_Net_Enhancing_ECCV_2018_paper.html)] [__`BNN`__] Bi-Real Net: Enhancing the Performance of 1-bit CNNs with Improved Representational Capability and Advanced Training Algorithm [[PyTorch, Coffee](https://github.com/liuzechun/Bi-Real-net)]

<details><summary>Bibtex</summary><pre><code>@InProceedings{Bi-Real-Net_ECCV_2018,
author = {Liu, Zechun and Wu, Baoyuan and Luo, Wenhan and Yang, Xin and Liu, Wei and Cheng, Kwang-Ting},
title = {{Bi-Real Net}: Enhancing the Performance of 1-bit CNNs with Improved Representational Capability and Advanced Training Algorithm},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}</code></pre></details>

[[ArXiv_2018](https://arxiv.org/abs/1812.01965)] [__`BNN+CPU/GPU`__] BMXNet-v2: Training Competitive Binary Neural Networks from Scratch [[BMXNet-v2](https://github.com/hpi-xnor/BMXNet-v2)]

<details><summary>Bibtex</summary><pre><code>@article{bmxnetv2,
  title = {Training Competitive Binary Neural Networks from Scratch},
  author = {Joseph Bethge and Marvin Bornstein and Adrian Loy and Haojin Yang and Christoph Meinel},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1812.01965},
  Year = {2018}
}</code></pre></details>

[[ICM_2017](https://arxiv.org/abs/1705.09864)] [__`BNN+CPU/GPU`__] BMXNet: An Open-Source Binary Neural Network Implementation Based on MXNet

<details><summary>Bibtex</summary><pre><code>@inproceedings{BMXNet_ICM_2017,
  title={Bmxnet: An open-source binary neural network implementation based on mxnet},
  author={Yang, Haojin and Fritzsche, Martin and Bartz, Christian and Meinel, Christoph},
  booktitle={Proceedings of the 25th ACM international conference on Multimedia},
  pages={1209--1212},
  year={2017}
}</code></pre></details>

[[NeurIPS_2016](https://proceedings.neurips.cc/paper/2016/hash/d8330f857a17c53d217014ee776bfd50-Abstract.html)] [__`BNN`__] Binarized Neural Networks

<details><summary>Bibtex</summary><pre><code>@incollection{BNN_NIPS_2016,
title = {Binarized Neural Networks},
author = {Hubara, Itay and Courbariaux, Matthieu and Soudry, Daniel and El-Yaniv, Ran and Bengio, Yoshua},
booktitle = {Advances in Neural Information Processing Systems 29},
pages = {4107--4115},
year = {2016},
publisher = {Curran Associates, Inc.}
}</code></pre></details>

[[ECCV_2016](https://arxiv.org/abs/1603.05279)] [__`BNN,BWN+CPU/GPU`__] XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks [[Torch7](https://github.com/allenai/XNOR-Net)] [[PyTorch](https://github.com/jiecaoyu/XNOR-Net-PyTorch)] 

<details><summary>Bibtex</summary><pre><code>@inproceedings{XNOR-Net_ECCV_2016,
  title={Xnor-net: Imagenet classification using binary convolutional neural networks},
  author={Rastegari, Mohammad and Ordonez, Vicente and Redmon, Joseph and Farhadi, Ali},
  booktitle={European conference on computer vision},
  pages={525--542},
  year={2016},
  organization={Springer}
}</code></pre></details>


------

### TNN

[[AAAI_2021](https://ojs.aaai.org/index.php/AAAI/article/view/17036)] [__`TNN`__] TRQ: Ternary Neural Networks With Residual Quantization

<details><summary>Bibtex</summary><pre><code>@inproceedings{TRQ_AAAI_2021,
  title={TRQ: Ternary Neural Networks With Residual Quantization},
  author={Li, Yue and Ding, Wenrui and Liu, Chunlei and Zhang, Baochang and Guo, Guodong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}</code></pre></details>

[[ICCV_2021](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_FATNN_Fast_and_Accurate_Ternary_Neural_Networks_ICCV_2021_paper.html)] [__`TNN`__] FATNN: Fast and Accurate Ternary Neural Networks [[PyTorch](https://github.com/zhuang-group/QTool)]

<details><summary>Bibtex</summary><pre><code>@InProceedings{FATNN_ICCV_2021,
    author    = {Chen, Peng and Zhuang, Bohan and Shen, Chunhua},
    title     = {FATNN: Fast and Accurate Ternary Neural Networks},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {5219-5228}
}</code></pre></details>

[[AAAI_2020](https://ojs.aaai.org/index.php/AAAI/article/view/5912)] [__`TNN+FPGA`__] RTN: Reparameterized Ternary Network

<details><summary>Bibtex</summary><pre><code>@article{RTN_AAAI_2020,
  title={RTN: Reparameterized Ternary Network},
  author={Li, Yuhang and Dong, Xin and Zhang, Sai Qian and Bai, Haoli and Chen, Yuanpeng and Wang, Wei},
  journal={The AAAI Conference on Artificial Intelligence (AAAI)},
  pages={4780-4787},
  year={2020}
}</code></pre></details>

[[NN_2018](https://arxiv.org/abs/1705.09283)] [__`TNN+FPGA`__] GXNOR-Net: Training deep neural networks with ternary weights and activations without full-precision memory under a unified discretization framework [[Theano](https://github.com/AcrossV/Gated-XNOR)]

<details><summary>Bibtex</summary><pre><code>@article{GXNOR-Net_NN_2018,
  title={GXNOR-Net: Training deep neural networks with ternary weights and activations without full-precision memory under a unified discretization framework},
  author={Deng, Lei and Jiao, Peng and Pei, Jing and Wu, Zhenzhi and Li, Guoqi},
  journal={Neural Networks},
  volume={100},
  pages={49--58},
  year={2018},
  publisher={Elsevier}
}</code></pre></details>

[[ECCV_2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Diwen_Wan_TBN_Convolutional_Neural_ECCV_2018_paper.html)] [__`TBN`__] TBN: Convolutional Neural Network with Ternary Inputs and Binary Weights [[PyTorch](https://github.com/dnvtmf/TBN)]

<details><summary>Bibtex</summary><pre><code>@inproceedings{TBN_ECCV_2018,
author = {Wan, Diwen and Shen, Fumin and Liu, Li and Zhu, Fan and Qin, Jie and Shao, Ling and Tao Shen, Heng},
title = {TBN: Convolutional Neural Network with Ternary Inputs and Binary Weights},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}</code></pre></details>

[[IJCNN_2017](https://arxiv.org/abs/1609.00222)] [__`TNN+FPGA`__] Ternary Neural Networks for Resource-Efficient AI Applications [[TNN-train](https://github.com/slide-lig/tnn-train)] [[TNN-convert](https://github.com/slide-lig/tnn_convert)] [[FPGA](http://tima.imag.fr/sls/project/ternarynn/)]

<details><summary>Bibtex</summary><pre><code>@inproceedings{TNN_IJCNN_2017,
  title={Ternary neural networks for resource-efficient AI applications},
  author={Alemdar, Hande and Leroy, Vincent and Prost-Boucle, Adrien and P{\'e}trot, Fr{\'e}d{\'e}ric},
  booktitle={2017 international joint conference on neural networks (IJCNN)},
  pages={2547--2554},
  year={2017},
  organization={IEEE}
}</code></pre></details>

[[ArXiv_2017](https://arxiv.org/abs/1705.01462)] [__`TNN`__] FGQ-TNN: Ternary Neural Networks with Fine-Grained Quantization

<details><summary>Bibtex</summary><pre><code>@article{FGQ-TTN_ArXiv_2017,
  title={Ternary neural networks with fine-grained quantization},
  author={Mellempudi, Naveen and Kundu, Abhisek and Mudigere, Dheevatsa and Das, Dipankar and Kaul, Bharat and Dubey, Pradeep},
  journal={arXiv preprint arXiv:1705.01462},
  year={2017}
}</code></pre></details>

------

### Mixed-Precision

[[Elec_2021](https://www.mdpi.com/2079-9292/10/8/886)] [__`mixed`__] Improving Model Capacity of Quantized Networks with Conditional Computation

<details><summary>Bibtex</summary><pre><code>@article{CC_Elec_2021,
  title={Improving Model Capacity of Quantized Networks with Conditional Computation},
  author={Pham, Phuoc and Chung, Jaeyong},
  journal={Electronics},
  volume={10},
  number={8},
  pages={886},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}</code></pre></details>

[[CVPR_2021](https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Liu_Layer_Importance_Estimation_With_Imprinting_for_Neural_Network_Quantization_CVPRW_2021_paper.html)] [__`mixed`__] Layer Importance Estimation With Imprinting for Neural Network Quantization 

<details><summary>Bibtex</summary><pre><code>@inproceedings{LIE_CVPR_2021,
  title={Layer Importance Estimation With Imprinting for Neural Network Quantization},
  author={Liu, Hongyang and Elkerdawy, Sara and Ray, Nilanjan and Elhoushi, Mostafa},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2408--2417},
  year={2021}
}</code></pre></details>

[[CVPR_2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_AQD_Towards_Accurate_Quantized_Object_Detection_CVPR_2021_paper.html)] [__`mixed`__] AQD: Towards Accurate Quantized Object Detection [[PyTorch](https://github.com/aim-uofa/model-quantization)]

<details><summary>Bibtex</summary><pre><code>@InProceedings{ADQ_CVPR_2021,
    author    = {Chen, Peng and Liu, Jing and Zhuang, Bohan and Tan, Mingkui and Shen, Chunhua},
    title     = {AQD: Towards Accurate Quantized Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {104-113}
}</code></pre></details>

[[Math_2021](https://www.mdpi.com/2227-7390/9/17/2144)] [__`mixed`__] NICE: Noise Injection and Clamping Estimation for Neural Network Quantization

<details><summary>Bibtex</summary><pre><code>@NICE_Math_2021,
AUTHOR = {Baskin, Chaim and Zheltonozhkii, Evgenii and Rozen, Tal and Liss, Natan and Chai, Yoav and Schwartz, Eli and Giryes, Raja and Bronstein, Alexander M. and Mendelson, Avi},
TITLE = {NICE: Noise Injection and Clamping Estimation for Neural Network Quantization},
JOURNAL = {Mathematics},
VOLUME = {9},
YEAR = {2021},
NUMBER = {17},
ARTICLE-NUMBER = {2144},
URL = {https://www.mdpi.com/2227-7390/9/17/2144},
ISSN = {2227-7390},
DOI = {10.3390/math9172144}
}</code></pre></details>

[[ICLR_2020](https://arxiv.org/abs/1902.08153)] [__`mixed`__] LSQ: Learned Step Size Quantization

<details><summary>Bibtex</summary><pre><code>@inproceedings{LSQ_ICLR_2020,
title={LEARNED STEP SIZE QUANTIZATION},
author={Steven K. Esser and Jeffrey L. McKinstry and Deepika Bablani and Rathinakumar Appuswamy and Dharmendra S. Modha},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=rkgO66VKDS}
}</code></pre></details>

[[ICLR_2020]()] [__`mixed`__] LLSQ: Linear Symmetric Quantization of Neural Networks for Low-precision Integer Hardware

<details><summary>Bibtex</summary><pre><code>@inproceedings{LLSQ_ICLR_2020,
title={Linear Symmetric Quantization of Neural Networks for Low-precision Integer Hardware},
author={Xiandong Zhao and Ying Wang and Xuyi Cai and Cheng Liu and Lei Zhang},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=H1lBj2VFPS}
}</code></pre></details>

[[CVPR_2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Jung_Learning_to_Quantize_Deep_Networks_by_Optimizing_Quantization_Intervals_With_CVPR_2019_paper.html)] [__`mixed`__] QIL: Learning to Quantize Deep Networks by Optimizing Quantization Intervals With Task Loss

<details><summary>Bibtex</summary><pre><code>@inproceedings{QIL_CVPR_2019,
  title={Learning to quantize deep networks by optimizing quantization intervals with task loss},
  author={Jung, Sangil and Son, Changyong and Lee, Seohyung and Son, Jinwoo and Han, Jae-Joon and Kwak, Youngjun and Hwang, Sung Ju and Choi, Changkyu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4350--4359},
  year={2019}
}</code></pre></details>

[[ArXiv_2019](https://arxiv.org/abs/1912.09356)] [__`mixed`__] FQ-Conv: Fully Quantized Convolution for Efficient and Accurate Inference

<details><summary>Bibtex</summary><pre><code>@article{FQ-Conv_Arxiv_2019,
  title={FQ-conv: Fully quantized convolution for efficient and accurate inference},
  author={Verhoef, Bram-Ernst and Laubeuf, Nathan and Cosemans, Stefan and Debacker, Peter and Papistas, Ioannis and Mallik, Arindam and Verkest, Diederik},
  journal={arXiv preprint arXiv:1912.09356},
  year={2019}
}</code></pre></details>

[[ECCV_2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Dongqing_Zhang_Optimized_Quantization_for_ECCV_2018_paper.html)] [__`mixed`__] LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks [[TensorPack](https://github.com/Microsoft/LQ-Nets)]

<details><summary>Bibtex</summary><pre><code>@inproceedings{LQ-Net_ECCV_2018,
  title={LQ-Nets: Learned quantization for highly accurate and compact deep neural networks},
  author={Zhang, Dongqing and Yang, Jiaolong and Ye, Dongqiangzi and Hua, Gang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={365--382},
  year={2018}
}</code></pre></details>

[[ArXiv_2018](https://arxiv.org/abs/1805.06085)] [__`QNN`__] PACT: Parameterized Clipping Activation for Quantized Neural Networks [[PyTorch](https://github.com/KwangHoonAn/PACT)]

<details><summary>Bibtex</summary><pre><code>@article{PACT_ArXiv_2018,
  title={Pact: Parameterized clipping activation for quantized neural networks},
  author={Choi, Jungwook and Wang, Zhuo and Venkataramani, Swagath and Chuang, Pierce I-Jen and Srinivasan, Vijayalakshmi and Gopalakrishnan, Kailash},
  journal={arXiv preprint arXiv:1805.06085},
  year={2018}
}</code></pre></details>

[[CVPR_2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Cai_Deep_Learning_With_CVPR_2017_paper.html)] [__`mixed`__] HWGQ: Deep Learning With Low Precision by Half-Wave Gaussian Quantization

<details><summary>Bibtex</summary><pre><code>@InProceedings{HWGQ_CVPR_2017,
author = {Cai, Zhaowei and He, Xiaodong and Sun, Jian and Vasconcelos, Nuno},
title = {Deep Learning With Low Precision by Half-Wave Gaussian Quantization},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}</code></pre></details>

[[ArXiv_2016](https://arxiv.org/abs/1606.06160)] [__`mixed+CPU/GPU`__] DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients [[TensorPack](https://github.com/tensorpack/tensorpack/tree/master/examples/DoReFa-Net)] 

<details><summary>Bibtex</summary><pre><code>@article{DoReFa-Net_ArXiv_2016,
  title={Dorefa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients},
  author={Zhou, Shuchang and Wu, Yuxin and Ni, Zekun and Zhou, Xinyu and Wen, He and Zou, Yuheng},
  journal={arXiv preprint arXiv:1606.06160},
  year={2016}
}</code></pre></details>

------


### INT8

[[AAAI_2021](https://ojs.aaai.org/index.php/AAAI/article/view/16462)] [__`INT8+GPU`__] Distribution Adaptive INT8 Quantization for Training CNNs

<details><summary>Bibtex</summary><pre><code>@inproceedings{DA-INT8_AAAI_2021,
  title={Distribution Adaptive INT8 Quantization for Training CNNs},
  author={Zhao, Kang and Huang, Sida and Pan, Pan and Li, Yinghan and Zhang, Yingya and Gu, Zhenyu and Xu, Yinghui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}</code></pre></details>

[[ArXiv_2020](https://arxiv.org/abs/2004.09602)] [__`INT8`__] Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation

<details><summary>Bibtex</summary><pre><code>@article{INT8_Nvidia_Arxiv_2020,
  title={Integer quantization for deep learning inference: Principles and empirical evaluation},
  author={Wu, Hao and Judd, Patrick and Zhang, Xiaojie and Isaev, Mikhail and Micikevicius, Paulius},
  journal={arXiv preprint arXiv:2004.09602},
  year={2020}
}</code></pre></details>

[[CVPR_2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_Towards_Unified_INT8_Training_for_Convolutional_Neural_Network_CVPR_2020_paper.html)] [__`INT8+GPU`__] UI8: Towards Unified INT8 Training for Convolutional Neural Network

<details><summary>Bibtex</summary><pre><code>@inproceedings{UINT8_CVPR_2020,
  title={Towards unified int8 training for convolutional neural network},
  author={Zhu, Feng and Gong, Ruihao and Yu, Fengwei and Liu, Xianglong and Wang, Yanfei and Li, Zhelong and Yang, Xiuqi and Yan, Junjie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1969--1979},
  year={2020}
}</code></pre></details>

------


### ImplementationAndAcceleration

[[ICPADS_2020](https://scholar.google.com/scholar?cluster=13318409564100224381&hl=zh-CN&as_sdt=0,5)] [__`CPU-BNN`__] XOR-Net: An Efficient Computation Pipeline for Binary Neural Network Inference on Edge Devices [[C++](https://github.com/yiweifengyan/XOR-Net)]

<details><summary>Bibtex</summary><pre><code>@inproceedings{XOR-Net_ICPADS_2020,
  title={XOR-Net: an efficient computation pipeline for binary neural network inference on edge devices},
  author={Zhu, Shien and Duong, Luan H. K. and Liu, Weichen},
  booktitle={The 26th IEEE International Conference on Parallel and Distributed Systems (ICPADS)},
  year={2020}
}</code></pre></details>

[[TVLSI_2020](https://arxiv.org/abs/1909.06892)] [__`IMC-TNN`__] TiM-DNN: Ternary In-Memory Accelerator for Deep Neural Networks

<details><summary>Bibtex</summary><pre><code>@article{TiM-DNN_TVLSI_2020,
  title={TiM-DNN: Ternary In-Memory Accelerator for Deep Neural Networks},
  author={Jain, Shubham and Gupta, Sumeet Kumar and Raghunathan, Anand},
  journal={IEEE Transactions on Very Large Scale Integration (VLSI) Systems},
  year={2020},
  publisher={IEEE}
}</code></pre></details>

[[MM_2019](https://arxiv.org/abs/1908.05858)] [__`CPU-BNN, ARM`__] daBNN: A Super Fast Inference Framework for Binary Neural Networks on ARM devices [[daBNN](https://github.com/JDAI-CV/dabnn)]

<details><summary>Bibtex</summary><pre><code>@inproceedings{daBNN_MM_2019,
  title={dabnn: A super fast inference framework for binary neural networks on arm devices},
  author={Zhang, Jianhao and Pan, Yingwei and Yao, Ting and Zhao, He and Mei, Tao},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={2272--2275},
  year={2019}
}</code></pre></details>

[[IPDPS_2018](https://scholar.google.com.sg/scholar?cluster=13767699997921028631&hl=en&as_sdt=0,5)] [__`CPU-BNN`__] BitFlow: Exploiting vector parallelism for binary neural networks on CPU

<details><summary>Bibtex</summary><pre><code>@inproceedings{BitFlow_2018_IPDPS,
  title={Bitflow: Exploiting vector parallelism for binary neural networks on cpu},
  author={Hu, Yuwei and Zhai, Jidong and Li, Dinghua and Gong, Yifan and Zhu, Yuhao and Liu, Wei and Su, Lei and Jin, Jiangming},
  booktitle={2018 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
  pages={244--253},
  year={2018},
  organization={IEEE}
}</code></pre></details>

[[TRETS_2018](https://hal.archives-ouvertes.fr/hal-01686718)] [__`FPGA-TNN`__] High-Efficiency Convolutional Ternary Neural Networks with Custom Adder Trees and Weight Compression [[FPGA](http://tima.imag.fr/sls/project/ternarynn/)]

<details><summary>Bibtex</summary><pre><code>@article{FPGA-TNN_TRETS_2018,
  TITLE = {{High-Efficiency Convolutional Ternary Neural Networks with Custom Adder Trees and Weight Compression}},
  AUTHOR = {Prost-Boucle, Adrien and BOURGE, Alban and P{\'e}trot, Fr{\'e}d{\'e}ric},
  URL = {https://hal.archives-ouvertes.fr/hal-01686718},
  JOURNAL = {{ACM Transactions on Reconfigurable Technology and Systems (TRETS)}},
  PUBLISHER = {{ACM}},
  SERIES = {Special Issue on Deep learning on FPGAs},
  VOLUME = {11},
  NUMBER = {3},
  PAGES = {1-24},
  YEAR = {2018},
  MONTH = Dec,
  DOI = {10.1145/3294768},
  PDF = {https://hal.archives-ouvertes.fr/hal-01686718v2/file/trets_nocopyright.pdf},
  HAL_ID = {hal-01686718},
  HAL_VERSION = {v2},
}</code></pre></details>

[[FPL_2017](https://hal.archives-ouvertes.fr/hal-01563763)] [__`FPGA-TNN`__] Scalable High-Performance Architecture for Convolutional Ternary Neural Networks on FPGA [[FPGA](http://tima.imag.fr/sls/project/ternarynn/)]

<details><summary>Bibtex</summary><pre><code>
@inproceedings{FPGA-TNN_FPL_2017,
  title={Scalable high-performance architecture for convolutional ternary neural networks on FPGA},
  author={Prost-Boucle, Adrien and Bourge, Alban and P{\'e}trot, Fr{\'e}d{\'e}ric and Alemdar, Hande and Caldwell, Nicholas and Leroy, Vincent},
  booktitle={2017 27th International Conference on Field Programmable Logic and Applications (FPL)},
  pages={1--7},
  year={2017},
  organization={IEEE}
}</code></pre></details>


[[]()] [__``__] 

<details><summary>Bibtex</summary><pre><code>
</code></pre></details>


