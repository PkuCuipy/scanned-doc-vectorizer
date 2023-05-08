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
"""

import fontforge
from pathlib import Path

font_folder = Path("./font/")
font_paths = font_folder.glob("*.*")
svg_output_folder = Path('./svg/')

for font_path in font_paths:
    font = fontforge.open(str(font_path), fstypepermitted_flag := 0x1)
    font.selection.all()
    for glyph in font.selection.byGlyphs:
        if not glyph.foreground.isEmpty():  # 空的图就不导出为 SVG 了
            glyph_filename = svg_output_folder / f'{font.fontname}__{glyph.encoding}__U+{glyph.unicode:04X}.svg'
            glyph.export(str(glyph_filename))
        # else:  # debug: 确保没导出的都是空的图
        #     glyph.export(str(svg_output_folder / f'~~~{font.fontname}__{glyph.encoding}__U+{glyph.unicode:04X}.svg'))
    print(font.fontname, "done.")
    font.close()
