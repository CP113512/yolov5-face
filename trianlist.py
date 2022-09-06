# yolov5基准
# python train.py --cfg models/hub_sly/yolov5s.yaml --device 1 --name s;
# python train.py --cfg models/hub_sly/yolov5s-transformer.yaml --device 0 --name s_c3tr;

# yolov5的4层检测
# python train.py --cfg models/hub_sly/yolov5s-4h.yaml --device 1 --name s_4h;
# python train.py --cfg models/hub_sly/yolov5s-4h-c3tr.yaml --device 1 --name s_4h_c3tr;

# yoloface基准
# python models/hub_sly/yolov5s-face.yaml --name s_face;
# python models/hub_sly/yolov5s-facev2.yaml --name s_facev2;
# python models/hub_sly/yolov5s-facev2-mul.yaml --name s_facev2_mul;

# 新卷积
# python train.py --cfg models/hub_sly/yolov5s-SPD-Conv.yaml --name s_GAMAttention;
# python train.py --cfg models/hub_sly/yolov5s-gnconv.yaml --name s_gnconv;

# 新结构-Hor
# python train.py --cfg models/hub_sly/yolov5s-HorBlock.yaml  --name s_HorBlock;
# python train.py --cfg models/hub_sly/yolov5s-HorNet.yaml --name s_HorNet;
# python train.py --cfg models/hub_sly/yolov5s-C3HB.yaml  --name s_C3HB;

# backbone结构修改
# python train.py --cfg models/hub_sly/yolov5s-BoTNet.yaml --device 0 --name s_BoTNet;
# python train.py --cfg models/hub_sly/yolov5s-CoTNet.yaml --device 0 --name s_CoTNet;
# python train.py --cfg models/hub_sly/yolov5s-PicoDet.yaml --device 1 --name s_PicoDet;

# ConvNeXt结合
# python train.py --cfg models/hub_sly/yolov5s-CNeB.yaml --device 1 --name s_CNeB;
# python train.py --cfg models/hub_sly/yolov5s-ConvNextBlock.yaml --device 1 --name s_ConvNextBlock;

# CRAFE算子
# python train.py --cfg models/hub_sly/yolov5s-CARAFE.yaml --name s_CARAFE;

# 新轻量网络
# python train.py --cfg models/hub_sly/yolov5s-MobileOneBlock.yaml --name s_MobileOneBlock;

# 加注意力
# python train.py --cfg models/hub_sly/yolov5s-GAMAttention.yaml --name s_GAMAttention;
# python train.py --cfg models/hub_sly/yolov5s-S2Attention.yaml --name s_S2Attention;
# python train.py --cfg models/hub_sly/yolov5s-SimAM.yaml --name s_SimAM;
# python train.py --cfg models/hub_sly/yolov5s-SKAttention.yaml --name s_SKAttention;
# python train.py --cfg models/hub_sly/yolov5s-SOCA.yaml --name s_SOCA;
# python train.py --cfg models/hub_sly/yolov5s-ACmix.yaml --name s_ACmix;


