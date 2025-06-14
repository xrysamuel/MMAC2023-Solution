import pandas as pd
from PIL import Image
import os
import colorsys
import numpy as np
from skimage.measure import shannon_entropy

def image_dataset_analyze(df: pd.DataFrame, output_dir: str):
    """
    分析图像数据集，并生成包含标签分布、色相、亮度、熵的 CSV 文件。

    参数:
    df (pd.DataFrame): 包含两列的 DataFrame。
                       'image' 列存放图像的路径。
                       'label' 列存放标签 (0~4)。
    output_dir (str): 用于存放输出 CSV 文件的目录。
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 标签分布 (label, count)
    label_distribution = df['label'].value_counts().reset_index()
    label_distribution.columns = ['label', 'count']
    label_distribution.to_csv(os.path.join(output_dir, 'label_distribution.csv'), index=False)
    print(f"标签分布已保存到: {os.path.join(output_dir, 'label_distribution.csv')}")

    # 初始化列表用于存储图像分析结果
    hue_list = []
    brightness_list = []
    entropy_list = []
    index_list = []
    label_list = []

    # 遍历 DataFrame 中的每一行，处理每张图片
    for index, row in df.iterrows():
        image_path = row['image']
        label = row['label']

        try:
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

        except FileNotFoundError:
            print(f"警告: 找不到图片文件: {image_path}，跳过此文件。")
            continue
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}，跳过此文件。")
            continue

    # 创建色相、亮度、熵的 DataFrame
    hue_df = pd.DataFrame({
        'index': index_list,
        'hue': hue_list,
        'label': label_list
    })
    hue_df.to_csv(os.path.join(output_dir, 'image_hue.csv'), index=False)
    print(f"每张图片的色相已保存到: {os.path.join(output_dir, 'image_hue.csv')}")

    brightness_df = pd.DataFrame({
        'index': index_list,
        'brightness': brightness_list,
        'label': label_list
    })
    brightness_df.to_csv(os.path.join(output_dir, 'image_brightness.csv'), index=False)
    print(f"每张图片的亮度已保存到: {os.path.join(output_dir, 'image_brightness.csv')}")

    entropy_df = pd.DataFrame({
        'index': index_list,
        'entropy': entropy_list,
        'label': label_list
    })
    entropy_df.to_csv(os.path.join(output_dir, 'image_entropy.csv'), index=False)
    print(f"每张图片的熵已保存到: {os.path.join(output_dir, 'image_entropy.csv')}")

    print("\n图像数据集分析完成。")
