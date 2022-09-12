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

# -----------------------------------跑！
# 注意力
# python train.py --cfg models/hub_sly2/yolov5s-4-CBAM.yaml --name s-4-CBAM;
# python train.py --cfg models/hub_sly2/yolov5s-gam.yaml --name s-gam;
# python train.py --cfg models/hub_sly2/yolov5s-coordAtt.yaml --name s-coordAtt;
# python train.py --cfg models/hub_sly2/yolov5s-c3tr.yaml --name s-c3tr;
# python train.py --cfg models/hub_sly2/yolov5s-c3str.yaml --name s-c3str;


# python train.py --cfg models/hub_sly2/yolov5s-4.yaml --name s-4
# python train.py --cfg models/hub_sly2/yolov5s-4-c3tr.yaml --name s-4-c3tr
# python train.py --cfg models/hub_sly2/yolov5s-4-c3tr-carafe.yaml --name s-4-c3tr-carafe

# 结构
# python train.py --cfg models/hub_sly2/yolov5s-4-CBAM-Swin-BiFPN.yaml --name s-4-CBAM-Swin-BiFPN;
# python train.py --cfg models/hub_sly2/yolov5s-4-CBAM-TPH-BiFPN.yaml --name s-4-CBAM-TPH-BiFPN;
# python train.py --cfg models/hub_sly2/yolov5s-bifpn.yaml --name s-bifpn;
# python train.py --cfg models/hub_sly2/yolov5s-bot.yaml --name s-bot;

# ----------------------------------------------下面的等明天，还在改

# 解耦头
# python train.py --cfg models/hub_sly2/yolov5s-asff.yaml --name s-asff;
# python train.py --cfg models/hub_sly2/yolov5s-asff-cbam.yaml --name s-asff-cbam;
# python train.py --cfg models/hub_sly2/yolov5s-4-asff.yaml --name s-4-asff;##############
# python train.py --cfg models/hub_sly2/yolov5s-decoupled.yaml --name s-decoupled;##############
# python train.py --cfg models/hub_sly2/yolov5s-4-decoupled.yaml --name s-4-decoupled;##############

# 卷积
# python train.py --cfg models/hub_sly2/yolov5s-involution.yaml --name s-involution;###########