# python train.py --cfg models\yolov5s_c3.yaml --device 0 --name yolov5s;


# (1)最新MobileOne结构换Backbone修改: yolov5_MobileOneBlock.yaml

# (2)Swin Transformer结构的修改:models/hub_sly/yolov5s-swin-transformer.yaml
# 出现RuntimeError: expected scalar type Half but found Float
# 解决办法：
# 1. train加个参数 使得opt.swin默认设置为false
# 2.val.run调用的时候加个(half=opt.swin)传进去，因为val.py默认的half为True，要将其设置为false

# (3)PicoDet结构的修改:models/hub_sly/yolov5s_PicoDet.yaml

# (4)CotNet Transformer结构的修改 models/hub_sly/yolov5s_CoTNet.yaml
# python train.py --cfg yolov5s_c3.yaml --device 0 --name yolov5s;

# (6)修改Soft-NMS,Soft-CIoUNMS,Soft-SIoUNMS,Soft-DIoUNMS,Soft-EIoUNMS,Soft-GIoUNMS
# 效果一般：在general.py文件中增加def soft_nms；
# 在val.py将out = non_max_suppression替换为out = soft_nms
# 在def soft_nms中，找到iou = bbox_iou替换为iou = bbox_iou(dc[0], dc[1:], CIoU=True)
# 训练时使用了DIOU，detect时也需要改为DIOU？

# (7)改进DIoU-NMS,SIoU-NMS,EIoU-NMS,CIoU-NMS,GIoU-NMS
# 1.改为：Merge-NMS:YOLOv5代码中直接打开即可,general.py文件下的merge = False替换为merge = True即可
# 2.改为：Soft-NMS
# 3.其他：NMS:在general.py文件中加入NMS方法其次
# 将non_max_suppression方法中的
# i = torchvision.ops.nms(boxes, scores, iou_thres),改为i = NMS(boxes, scores, iou_thres, class_nms='xxx')class_nms='DIoU'

# (8)增加ACmix结构的修改,自注意力和卷积集成:models/hub_sly/yolov5s-ACmix.yaml
# 出现RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same
# 解决办法：1.train加个参数parser.add_argument('--acmix', action='store_true', help='useacmix')
# 2.val.run调用的时候加个(half=not opt.acmix)传进去，因为val.py默认的half为True，要将其设置为false。

# (9)BoTNet Transformer结构的修改:models/hub_sly/yolov5s-BoTNet.yaml
# (10)最新HorNet结合YOLO应用首发！ | ECCV2022出品，多种搭配，即插即用 | Backbone主干、递归门控卷积的高效高阶空间交互
# 1.在YOLOv5中 使用 gnconv模块 示例 models/hub_sly/yolov5s-gnconv.yaml
# 2.在YOLOv5中 使用 HorBlock模块 示例 models/hub_sly/yolov5s-HorBlock.yaml
# 3.在YOLOv5中 使用 HorNet主干网络 示例 models/hub_sly/yolov5s-HorNet.yaml
# 4.在YOLOv5中 使用 新增C3HB结构 示例 models/hub_sly/yolov5s-C3HB.yaml

# (10) ConvNeXt结合YOLO | CVPR2022 多种搭配，即插即用 | Backbone主干CNN模型
# 1.在YOLOv5中 使用 CNeB 模块 示例 models/hub_sly/yolov5s-CNeB.yaml
# 2.在YOLOv5中 使用 ConvNeXtBlock 模块 示例 models/hub_sly/yolov5s-ConvNextBlock.yaml