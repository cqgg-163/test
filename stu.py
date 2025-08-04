import re
import os


def clean_log_file(input_path):
    # 确定输出路径（在同一目录下命名为ZRZ）
    output_path = os.path.join(os.path.dirname(input_path), "ZRZ.log")

    # 编译正则表达式用于匹配需要删除的内容
    bracket_pattern = re.compile(r'\[.*?\]')  # 匹配中括号及其内容
    time_pattern = re.compile(
        r'data_time: [\d\.]+\([\d\.]+\) '
        r'iter_time: [\d\.]+\([\d\.]+\) '
        r'remain_time: \d{2}:\d{2}:\d{2}:\d{2}'
    )  # 匹配时间信息模式

    # 中文字符到英文字符的映射表
    chinese_to_english = {
        '，': ',', '。': '.', '；': ';', '：': ':',
        '？': '?', '！': '!', '（': '(', '）': ')',
        '【': '[', '】': ']', '「': '{', '」': '}',
        '《': '<', '》': '>', '、': '/', '“': '"',
        '”': '"', '‘': "'", '’': "'", '￥': '$',
        '。': '.', '·': '`', '—': '-', '～': '~'
    }

    with open(input_path, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            # 删除中括号及其内容
            cleaned_line = bracket_pattern.sub('', line)

            # 删除时间信息模式
            cleaned_line = time_pattern.sub('', cleaned_line)

            # 替换中文字符为英文字符
            for cn_char, en_char in chinese_to_english.items():
                cleaned_line = cleaned_line.replace(cn_char, en_char)

            # 写入处理后的行
            outfile.write(cleaned_line)

    print(f"File processing completed! Saved to: {output_path}")
    print(f"Original file size: {os.path.getsize(input_path)} bytes")
    print(f"New file size: {os.path.getsize(output_path)} bytes")


# 使用示例
if __name__ == "__main__":
    log_file_path = "/home/zxb/GuidedContrastrandnew/exp/s3dis/semantic_semi/semseg_run1_100_baseline_s3dis/train-20250727_025944.log"
    clean_log_file(log_file_path)