from PIL import Image
import colorsys

def generate_grayscale_color_bar(width, height, start_lightness, end_lightness, filename="grayscale_bar.png"):
    """
    生成一个HSL空间中L值变化的灰色色彩条带图片。

    Args:
        width (int): 图片宽度。
        height (int): 图片高度。
        start_lightness (float): 起始亮度值 (0.0-1.0)。
        end_lightness (float): 结束亮度值 (0.0-1.0)。
        filename (str): 输出文件名。
    """
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    # Hue 和 Saturation 对于灰色条带是固定的
    fixed_hue = 0.0  # 任意Hue值，因为饱和度为0，所以无关紧要
    fixed_saturation = 0.0 # 关键：饱和度为0表示灰色

    for x in range(width):
        # 计算当前像素的Lightness值
        current_lightness = start_lightness + (end_lightness - start_lightness) * (x / (width - 1))
        
        # 将HLS转换为RGB
        # colorsys的hls_to_rgb期望h, l, s都在0.0到1.0之间
        r, g, b = colorsys.hls_to_rgb(fixed_hue, current_lightness, fixed_saturation)
        
        # 将RGB值从0.0-1.0转换为0-255
        r_int = int(r * 255)
        g_int = int(g * 255)
        b_int = int(b * 255)
        
        # 填充当前列的像素
        for y in range(height):
            pixels[x, y] = (r_int, g_int, b_int)
    
    img.save(filename)
    print(f"灰色色彩条带已保存为: {filename}")

# 调用函数生成L从0到0.6的灰色色彩条带
generate_grayscale_color_bar(
    width=600,  # 图片宽度
    height=100, # 图片高度
    start_lightness=0.0,
    end_lightness=0.6
)