# 2023-03-25

import fitz
from PIL import Image

doc = fitz.Document("../data/2.pdf")

print(doc.page_count)   # PDF 总页数

for i in range(min(doc.page_count, 3)):

    page = doc.load_page(i)

    pix = page.get_pixmap(dpi=300, colorspace="gray")       # 如果是彩色就换成 "rgb"
    pix.save(f"./result/{i}.png")

    # img = Image.frombytes("L", (pix.width, pix.height), pix.samples)    # 如果是彩色就换成 "RGB"
    # img.show()

    svg = page.get_svg_image(matrix=fitz.Identity)
    print(svg, file=open(f"./result/{i}.svg", "w"))




doc.close()