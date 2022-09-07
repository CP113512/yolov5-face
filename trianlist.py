# yolov5基准
# python train.py --cfg models/hub_sly/yolov5s.yaml --name s --device 0;
# python train.py --cfg models/hub_sly/yolov5s-transformer.yaml --name s_c3tr --device 0;

# yolov5的4层检测
# python train.py --cfg models/hub_sly/yolov5s-4h.yaml --device 0 --name s_4h --device 0;
# python train.py --cfg models/hub_sly/yolov5s-4h-c3tr.yaml --device 0 --name s_4h_c3tr --device 0;

# yoloface基准
# python train.py --cfg models/hub_sly/yolov5s-face.yaml --name s_face;
# python train.py --cfg models/hub_sly/yolov5s-facev2.yaml --name s_facev2;######################
# python train.py --cfg models/hub_sly/yolov5s-facev2-mul.yaml --name s_facev2_mul;#######################

# 新卷积
# python train.py --cfg models/hub_sly/yolov5s-SPD-Conv.yaml --name s_GAMAttention --device 0;
# python train.py --cfg models/hub_sly/yolov5s-gnconv.yaml --name s_gnconv --device 0;

# 新结构-Hor
# python train.py --cfg models/hub_sly/yolov5s-HorBlock.yaml  --name s_HorBlock --device 0;
# python train.py --cfg models/hub_sly/yolov5s-HorNet.yaml --name s_HorNet;
# python train.py --cfg models/hub_sly/yolov5s-C3HB.yaml  --name s_C3HB --device 0;

# backbone结构修改
# python train.py --cfg models/hub_sly/yolov5s-BoTNet.yaml --name s_BoTNet --device 0;
# python train.py --cfg models/hub_sly/yolov5s-CoTNet.yaml --name s_CoTNet --device 0;
# python train.py --cfg models/hub_sly/yolov5s-PicoDet.yaml --name s_PicoDet --device 0;

# ConvNeXt结合
# python train.py --cfg models/hub_sly/yolov5s-CNeB.yaml --device 0 --name s_CNeB --device 0;
# python train.py --cfg models/hub_sly/yolov5s-ConvNextBlock.yaml --device 0 --name s_ConvNextBlock --device 0;
###################################################################################
# CRAFE算子
# python train.py --cfg models/hub_sly/yolov5s-CARAFE.yaml --name s_CARAFE --device 0;

# 新轻量网络
# python train.py --cfg models/hub_sly/yolov5s-MobileOneBlock.yaml --name s_MobileOneBlock --device 0;

# 加注意力
# python train.py --cfg models/hub_sly/yolov5s-GAMAttention.yaml --name s_GAMAttention --device 0;
# python train.py --cfg models/hub_sly/yolov5s-S2Attention.yaml --name s_S2Attention --device 0;
# python train.py --cfg models/hub_sly/yolov5s-SimAM.yaml --name s_SimAM --device 0;
# python train.py --cfg models/hub_sly/yolov5s-SKAttention.yaml --name s_SKAttention --device 0;
# python train.py --cfg models/hub_sly/yolov5s-SOCA.yaml --name s_SOCA --device 0;
# python train.py --cfg models/hub_sly/yolov5s-ACmix.yaml --name s_ACmix --device 0;#########################


