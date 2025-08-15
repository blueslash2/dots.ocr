dict_promptmode_to_prompt = {
    # prompt_layout_all_en: parse all layout info in json format.
    "完全识别": """输出来自图像的布局信息，参考'布局类别'把图像分区块分解成多个布局元素bbox，以及bbox的类别category，以及识别bbox内对应的文本内容字段text。

Bbox以左上、右下坐标点的坐标格式表示: [x1, y1, x2, y2]
布局类别: 可选类别为 ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
**每个bbox输出样例**： { 'bbox': [19, 18, 342, 40], 'category': 'Text', 'text': '账户信息\\\n客户信息......' }, { 'bbox': [ 19, 44, 342, 88], ... } ...

文本提取与格式规则:
- Picture: 对于 'Picture' 类别，不输出text字段。
- Formula: 将其文本格式化为LaTeX。
- Table: 将表格整体其连同文本格式化为HTML结构（利用table thead tr th tbody td等标签），不要把每个单元格单独拆分成bbox。
- 其它所有(Text、Title 等): 将其文本格式化为Markdown。

约束:
- 输出文本必须是图像中的原始文本，不得翻译。
- 所有布局元素必须按照人类的阅读顺序排序。
- 最终输出: 整个输出必须是一个单一、整体的JSON格式数据对象，在首尾"["、"]"以外不要输出多余字符。
""",

    # prompt_layout_only_en: layout detection
    "布局识别": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",

    # prompt_layout_only_en: parse ocr text except the Page-header and Page-footer
    "文字识别": """Extract the text content from this image.""",

    # prompt_grounding_ocr: extract text content in the given bounding box
    "选定文字识别": """Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).\nBounding Box:\n""",

    # "prompt_table_html": """Convert the table in this image to HTML.""",
    # "prompt_table_latex": """Convert the table in this image to LaTeX.""",
    # "prompt_formula_latex": """Convert the formula in this image to LaTeX.""",
}
