# 2023-03-26(00.17.25)
# 看黄楠 pyq 推荐 cursor, 用了一下, woc 直接写出来一个能 run 的..
# 2023-03-26(23.14.51)
# Cuipy 基于对 Copilot 的 prompt 进行重写

import cv2
import svgwrite
import matplotlib.pyplot as plt

image_path = "data/test_5.png"
output_path = "./test_5.svg"

# Read image and convert to grayscale
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binarize image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours with hierarchy
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create SVG drawing
dwg = svgwrite.Drawing(output_path, profile='tiny')

# Iterate over contours
for contour in contours:
    # Get coordinates of contour
    points = contour[:, 0, :]

    # Create polygon from contour
    polygon = dwg.polygon(points.tolist(), fill='black')

    # Add polygon to SVG drawing
    dwg.add(polygon)

# Save SVG drawing
dwg.save()

print("done")