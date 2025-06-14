from PIL import Image
import colorsys

def generate_hue_color_bar(width, height, start_hue, end_hue, lightness=0.5, saturation=1.0, filename="color_bar.png"):
    """
    生成一个HLS色彩条带图片。

    Args:
        width (int): 图片宽度。
        height (int): 图片高度。
        start_hue (float): 起始Hue值 (0-360)。
        end_hue (float): 结束Hue值 (0-360)。
        lightness (float): 亮度值 (0.0-1.0)。
        saturation (float): 饱和度值 (0.0-1.0)。
        filename (str): 输出文件名。
    """
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    for x in range(width):
        # 计算当前像素的Hue值
        # 将Hue值映射到0-1的范围，因为colorsys期望这个范围
        current_hue = start_hue + (end_hue - start_hue) * (x / (width - 1))
        
        # 将HLS转换为RGB
        # colorsys的hls_to_rgb期望h, l, s都在0.0到1.0之间
        r, g, b = colorsys.hls_to_rgb(current_hue / 360.0, lightness, saturation)
        
        # 将RGB值从0.0-1.0转换为0-255
        r_int = int(r * 255)
        g_int = int(g * 255)
        b_int = int(b * 255)
        
        # 填充当前列的像素
        for y in range(height):
            pixels[x, y] = (r_int, g_int, b_int)
    
    img.save(filename)
    print(f"色彩条带已保存为: {filename}")

# 调用函数生成Hue为4到44的色彩条带
generate_hue_color_bar(
    width=600,  # 图片宽度
    height=100, # 图片高度
    start_hue=4,
    end_hue=44,
    lightness=0.5, # 默认亮度为50%
    saturation=1.0  # 默认饱和度为100%
)