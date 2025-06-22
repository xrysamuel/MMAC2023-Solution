一、背景

病理性近视黄斑病变是高度近视患者视力丧失的主要原因
 （Silva, 2012; Yokoi and Ohno-Matsui, 2018）
大规模筛查依靠人工判读费时费力
META-PM 分级（5 类）：基于对患者眼底照片的分析

无近视性视网膜退行性病变
仅有豹纹状眼底
弥漫性脉络膜视网膜萎缩
斑片状脉络膜视网膜萎缩
黄斑萎缩

二、概述：两方面 框架 + 实验

多策略框架：
- 基于神经网络的分类器
- 数据增强流水线
- 测试时适应 TTA 方法：TENT
- 模型融合方法：Greedy Model Soup
实验：数据分析，模型横向对比，大规模预训练模型实验，消融实验

三、基于神经网络的分类器

介绍 ResNet，介绍 BasicBottle（resnet 18） 和 BottleNeck（resnet 50），介绍 BatchNorm

介绍 MobileNet，介绍什么叫做深度可分离的卷积

介绍 EfficientNet，介绍什么叫做 Compound Scaling，与其他 Scaling 的对比

介绍 ViT，介绍 Pathify 和 embedding

介绍 ConvNeXt，如何使用层次化的 encoder 和 decoder 架构，以及如何使用 GELU 和 LN

四、数据增强流水线

CutMix：随机从另一张图片上截取一个 patch 贴到当前图片上，真实标签概率分布相应混合，这里使用原位
随机翻转，小角度旋转，高斯模糊
随机亮度对比度调节，随机 HSV 抖动
Label Smoothing

这一块结合实验结果讲一讲为什么这么设计？

五、TTA 方法

TTA 为什么重要？
在框架中，实现了 TENT 方法
TENT 方法是什么？

六、Ensemble 方法

Ensemble 为什么重要？
在框架中，实现了 Greedy Soup 方法


七、训练实验

大概解读一下训练结果，然后引出一个结论——数据受限，一味精进模型结构没有意义

八、消融实验

数据增强

TENT

数据受限，所以围绕数据和非训练阶段的技巧会带来实质的提升

九、总结与不足



