import argparse
from typing import List
import re
from collections import Counter
import numpy as np
def preprocess_text(text: str) -> List[str]:
    """
    将输入的字符串进行预处理，返回一个预处理后的词列表。
    参数:
    text (str): 输入的字符串。
    返回:
    List[str]: 预处理后的词列表。
    """
    # 移除所有标点符号、数字和特殊字符
    text = re.sub(r'[^\u4e00-\u9fff]', '', text)
    # 将文本分割成词（在本例中，词是指单个字符）
    words = list(text)
    return words
def calculate_cosine_similarity(doc1_words: List[str], doc2_words: List[str]) -> float:
    """
    计算两个文档之间的余弦相似度。
    参数:
    doc1_words (List[str]): 文档1的词列表。
    doc2_words (List[str]): 文档2的词列表。
    返回:
    float: 两个文档之间的余弦相似度。
    """
    # 创建文档1和文档2的词频字典
    doc1_word_freq = Counter(doc1_words)
    doc2_word_freq = Counter(doc2_words)
    # 获取所有词的集合
    all_words = set(doc1_word_freq.keys()).union(set(doc2_word_freq.keys()))
    # 创建词向量
    doc1_vector = np.array([doc1_word_freq.get(word, 0) for word in all_words])
    doc2_vector = np.array([doc2_word_freq.get(word, 0) for word in all_words])
    # 计算余弦相似度
    cosine_similarity = np.dot(doc1_vector, doc2_vector) / (np.linalg.norm(doc1_vector) * np.linalg.norm(doc2_vector))
    return cosine_similarity
def main(orig_file_path, copy_file_path, output_file_path):
    # 读取原文和抄袭版论文的内容
    with open(orig_file_path, 'r', encoding='gbk') as orig_file, open(copy_file_path, 'r', encoding='gbk') as copy_file:
        orig_text = orig_file.read()
        copy_text = copy_file.read()
    # 预处理文本
    orig_words = preprocess_text(orig_text)
    copy_words = preprocess_text(copy_text)
    # 计算余弦相似度
    similarity_rate = calculate_cosine_similarity(orig_words, copy_words)
    # 将相似度写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(f"{similarity_rate:.2f}")
if __name__ == "__main__":
    # 初始化解析器
    parser = argparse.ArgumentParser(description="计算两个文档之间的相似度。")
    # 添加命令行参数
    parser.add_argument("orig_file_path", help="原文文件的绝对路径。")
    parser.add_argument("copy_file_path", help="抄袭版论文文件的绝对路径。")
    parser.add_argument("output_file_path", help="输出答案文件的绝对路径。")
    # 解析参数
    args = parser.parse_args()
    # 使用提供的文件路径调用主函数
    main(args.orig_file_path, args.copy_file_path, args.output_file_path)