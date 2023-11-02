一、Introduction
This project is finished based on the AutoDL, mmdetection, mmyolo and PaddleDetection. For the specific configurations, please refer to the following official documents:
mmdetection：https://mmdetection.readthedocs.io/en/latest/
mmyolo：https://mmyolo.readthedocs.io/en/latest/
PaddleDetection：https://hub.njuu.cf/PaddlePaddle/PaddleDetection/tree/release/2.6/docs
AutoDL:https://www.autodl.com/docs/

If you use mmdetection in your research, please cite this project.
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}

mmdetection contains: FCOS、yolox、yolov3、Retinanet、Faster-RCNN、DETR、yolov5
PaddleDetection contains: SSD

For instance, the configuration of mmdetection is as follows:

System environment:
    sys.platform: linux
    Python: 3.8.17 (default, Jul  5 2023, 21:04:15) [GCC 11.2.0]
    CUDA available: True
    numpy_random_seed: 77975333
    GPU 0: NVIDIA GeForce RTX 2080 Ti
    CUDA_HOME: /usr/local/cuda
    NVCC: Cuda compilation tools, release 11.6, V11.6.124
    GCC: gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
    PyTorch: 2.0.1+cu117
    PyTorch compiling details: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201703
  - Intel(R) oneAPI Math Kernel Library Version 2022.2-Product Build 20220804 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.7.3 (Git Hash 6dbeffbae1f23cbbeae17adb7b5b13f1f37c080e)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
  - CuDNN 8.5
  - Magma 2.6.1
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=0 -fabi-version=11 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wunused-local-typedefs -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.0.1, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, 

    TorchVision: 0.15.2+cu117
    OpenCV: 4.8.0
    MMEngine: 0.8.4

Runtime environment:
    cudnn_benchmark: False
    dist_cfg: {'backend': 'nccl'}
    mp_cfg: {'mp_start_method': 'fork', 'opencv_num_threads': 0}
    seed: 77975333
    Distributed launcher: none
    Distributed training: False
    GPU number: 1

二、MVDD13 dataset

├── MVDD13: root
	├── train: training set (25541)
	├── val: validation set (2838)
        ├── test: testing set (7095)
	└── annotations: label set
		├── instances_train.json		
		├── instances_val.json			
		├── instances_test.json			

三、Training and testing process on MVDD13
		
Take running yolox in mmdetection on Autodl as an example, after finishing the configuration according to the relevant official documents of mmdetection, you can run yolox-MVDD according to the following commands:
1.categories：autodl-tmp/mmdetection-main/mmdet/datasets/coco.py

2.class name：在autodl-tmp/mmdetection-main/mmdet/evaluation/functional/class_names.py

3.training MVDD13:python autodl-tmp/mmdetection-main/tools/train.py \
            >>autodl-tmp/mmdetection-main/mmdetection-main/configs/yolox-MVDD/yolox_s_8xb8-200e_coco-MVDD.py \
            >>--work-dir yolox

4.testing MVDD13:python autodl-tmp/mmdetection-main/tools/test.py \
            >>autodl-tmp/mmdetection-main/mmdetection-main/configs/yolox-MVDD/yolox_s_8xb8-200e_coco-MVDD.py \
            >>autodl-tmp/mmdetection-main/yolox20230823/best_coco_bbox_mAP_epoch_190.pth
	    >>--cfg-options test_evaluator.classwise=True
            >>--work-dir autodl-tmp/mmdetection-main/yolox \
	    >>--out autodl-tmp/mmdetection-main/yolox.pkl \
            >>--show

