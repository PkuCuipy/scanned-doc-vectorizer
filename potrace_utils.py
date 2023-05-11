# 2023-04-08(19.51.29)
# credit: Copilot & ChatGPT

"""
如果使用命令行:
PNG_FILENAME="test_6.png"
convert "./${PNG_FILENAME}.png" "${PNG_FILENAME}.ppm"
potrace "${PNG_FILENAME}.ppm" -s -o "${PNG_FILENAME}.svg"
"""


import subprocess
from PIL import Image
import io
import cv2
import regex as re


# 传入二维矩阵 (0-255), 调用 potrace 勾勒为 <svg> 后返回 <path> 对应的字符串
transform_pattern = re.compile(r'(?<=transform=")[^"]+')
path_data_pattern = re.compile(r'(?<=<path d=")[^"]+')

def mat_to_g_elem(mat):
    im = Image.fromarray(mat.astype("u1"))  # Convert numpy array to PIL image
    with io.BytesIO() as f:                 # Save PIL image to temporary file `io.BytesIO()`
        im.save(f, format='PPM')
        f.seek(0)
        ppm_data = f.read()
    # Run potrace on the temporary .PPM file
    potrace_process = subprocess.Popen(['potrace', '-s', '--flat', '-o', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    svg_data, stderr = potrace_process.communicate(input=ppm_data)
    data_str = svg_data.decode()
    d = "" if (result := path_data_pattern.findall(data_str)) == [] else result[0].replace("\n", " ")
    transform = transform_pattern.search(data_str).group()
    return {
        "d": d,
        "transform": transform
    }

if __name__ == "__main__":
    mat = cv2.imread("data/test_16_single_h.png", cv2.IMREAD_GRAYSCALE)
    print(mat_to_g_elem(mat))
