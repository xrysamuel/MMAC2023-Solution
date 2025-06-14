现在，请你写实验部分

首先是第一部分，数据分析

我给你大概的结果，你从中抽取值得说的信息，然后组成学术文章

首先一个结果是 Label 分布，在训练集熵统计了一下 Label 分布，发现标签为 Tessellated Fundus 和 No Macular Lesions 的比较多，Diffuse Chorioretinal Atrophy 和 Patchy Chorioretinal Atrophy 和 Macular Atrophy 比较少，见图 1，有潜在的类别不平衡风险。

这段代码给你参考一下，在文章里先简单说说各个指标如何测出来的

```python
with Image.open(image_path) as img:
    # 转换为 RGB 模式以确保兼容性
    img_rgb = img.convert("RGB")

    # 2. 每张图片的色相 (index, hue, label)
    # 计算平均色相
    # 色相的计算通常基于 RGB 颜色空间转换到 HSL 或 HSV。
    # 我们可以遍历每个像素，计算其色相，然后求平均。
    # 这里为了简化，我们计算图片的平均 RGB 值，然后转换。
    # 注意：对于单色或色相变化很大的图片，这种平均值可能不代表整体色相。
    # 更精确的方法是对每个像素计算色相，然后统计其分布或众数。
    # 为了满足“每张图片的色相”，我们使用平均色相。
    r, g, b = 0, 0, 0
    pixels = img_rgb.getdata()
    num_pixels = len(pixels)
    for pixel_r, pixel_g, pixel_b in pixels:
        r += pixel_r
        g += pixel_g
        b += pixel_b
    avg_r = r / num_pixels / 255.0  # 归一化到 0-1
    avg_g = g / num_pixels / 255.0
    avg_b = b / num_pixels / 255.0

    # 将 RGB 转换为 HSL
    h, l, s = colorsys.rgb_to_hls(avg_r, avg_g, avg_b)
    hue_list.append(h * 360)  # 色相通常表示为 0-360 度

    # 3. 每张图片的亮度 (index, brightness, label)
    # 亮度（Luminance）可以通过多种方式计算，这里使用 HSL 的 L 分量。
    # 另一种常见的亮度计算方法是 (R*0.299 + G*0.587 + B*0.114)。
    # 这里我们使用 HSL 的 L 作为亮度。
    brightness_list.append(l)

    # 4. 每张图片的熵 (index, entropy, label)
    # 计算图像熵需要灰度图像
    img_gray = img.convert("L")  # 转换为灰度图
    image_array = np.array(img_gray)
    entropy = shannon_entropy(image_array)
    entropy_list.append(entropy)

    index_list.append(index)
    label_list.append(label)
```

然后是 Brightness 分布，图像的香农熵分布和 Hue 分布，我们统计了它们在训练集上的不同标签上的分布，以及在整个训练集上的分布和在验证集熵的分布。

首先，在总体意义上，这些眼底图像的亮度的分布范围较广，不同样本之间差异显著，熵和 hue 的分布范围较窄，其中 hue 大致集中在黄色到红色范围内。

然后，我们发现在不同类别上，这三个指标都有明显的平均值差异。

最后，我们发现，这三个指标在训练集和验证集上的分布大致相同。