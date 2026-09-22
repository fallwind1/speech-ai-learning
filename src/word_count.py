from pathlib import Path
import re
import json
import sys

def is_chinese_char(char):
    return "\u4e00" <= char <= "\u9fff"

def count_text(text):
    chinese_count = 0
    chinese_freq = {}

    for char in text:
        if is_chinese_char(char):
            chinese_count += 1
            chinese_freq[char] = chinese_freq.get(char,0) + 1

    english_words = re.findall(r"[A-Za-z]+",text)
    english_freq = {}
    for word in english_words:
        word.lower()
        english_freq[word] = english_freq.get(word,0) + 1

    result = {
        "chinese_count":chinese_count,
        "chinese_freq":chinese_freq,
        "english_count":len(english_words),
        "english_freq":english_freq
    }
    return result

def process_file(input_file, output_file):
    """读取文本文件，统计并保存JSON结果"""
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"错误：找不到输入文件：{input_file}")
        return

    text = input_path.read_text(encoding="utf-8")

    #统计
    result = count_text(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    #保存JSON
    with output_path.open("w",encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )
        print(f"输入文件：{input_path}")
        print(f"输出文件：{output_path}")
        print("统计完成：")
        print(json.dumps(result, ensure_ascii=False, indent=2))

def main():

    if len(sys.argv) != 3:
        print("使用方法：")
        print(
            r"Python src\word_count.py <输入文件> <输出JSON文件>"
        )
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    process_file(input_file,output_file)

if __name__ == "__main__":
    main()