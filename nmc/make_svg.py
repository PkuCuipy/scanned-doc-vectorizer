#!/opt/homebrew/bin/fontforge
"""
2023-05-08
Desc: 将 `./font/` 文件夹中的所有字体文件转为 SVG 文件, 存储在 `./svg/` 文件夹中
Run: $> fontforge -script make_svg.py
Ref:
- https://fontforge.org/docs/scripting/python.html
- https://fontforge.org/docs/scripting/python/fontforge.html
- FSType: https://fontforge.org/docs/faq.html
- fstypepermitted: https://fontforge.org/docs/scripting/python/fontforge.html
- Unicode区段: https://zh.wikipedia.org/wiki/Unicode%E5%8D%80%E6%AE%B5
"""

import fontforge
from pathlib import Path
import random

font_folder = Path("./font/")
font_paths = font_folder.glob("*.*")
svg_output_folder = Path('./svg/')

for font_path in font_paths:
    print(list(fontforge.fontsInFile(str(font_path))))
    for sub_font in fontforge.fontsInFile(str(font_path)):
        print(f"\n\n读取 {font_path}({sub_font}):")
        font = fontforge.open(f"{font_path}({sub_font})", fstypepermitted_flag := 0x1)

        # ASCII 部分全都要
        font.selection.select(("ranges", None), 0, 0x7F)
        ascii_glyphs = list(font.selection.byGlyphs)
        for glyph in ascii_glyphs:
            if not glyph.foreground.isEmpty():  # 空的图就不导出为 SVG 了
                glyph_filename = svg_output_folder / f'{font.fontname}__{glyph.encoding}__U+{glyph.unicode:04X}.svg'
                glyph.export(str(glyph_filename))
        print(f"输出了 {len(ascii_glyphs)} 个 ASCII 字符")

        # 其他部分随机抽取 nr_samples 个
        nr_samples = 10
        font.selection.invert()
        else_glyphs = list(font.selection.byGlyphs)
        else_glyphs = random.sample(else_glyphs, k=min(nr_samples, len(else_glyphs)))
        for glyph in else_glyphs:
            if not glyph.foreground.isEmpty():  # 空的图就不导出为 SVG 了
                glyph_filename = svg_output_folder / f'{font.fontname}__{glyph.encoding}__U+{glyph.unicode:04X}.svg'
                glyph.export(str(glyph_filename))
        print(f"输出了 {len(else_glyphs)} 个其他字符")

        print(font.fontname, "done.")
        font.close()
