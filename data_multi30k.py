"""
这个脚本主要是使用moses分词，相当于我们用正则表达式插入空格等一系列预处理操作
- 将标点符号与单词分离（如"Hello, world!" → "Hello , world !"）
- 处理缩写（如"I'm" → "I 'm"，"can't" → "ca n't"）
"""
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
from sacremoses import MosesTokenizer
from pathlib import Path
import argparse


def moses_cut(in_file, out_file, lang):
    mt = MosesTokenizer(lang=lang) # 初始化分词器
    out_f = open(out_file, "w", encoding="utf8")
    with open(in_file, "r", encoding="utf8") as f:
        for line in f.readlines():#每读取一行，进行分词，并写入一行到新的文件中
            line = line.strip()
            if not line:
                continue
            # 7. 使用Moses分词器对当前行进行分词
            #    return_str=True: 返回字符串而不是token列表
            #    分词操作包括：
            #    - 将标点符号与单词分离（如"Hello, world!" → "Hello , world !"）
            #    - 处理缩写（如"I'm" → "I 'm"，"can't" → "ca n't"）
            #    - 分离引号、括号等特殊字符
            #    - 处理货币符号、百分比等
            #    - 根据语言特定规则进行分词
            cut_line = mt.tokenize(line, return_str=True) # 分词
            # 8. 将分词结果转为小写，并写入输出文件
            #    注意：转为小写可能会丢失专有名词的大小写信息
            #    如果任务需要保留大小写，可以移除.lower()调用
            out_f.write(cut_line.lower() + "\n") #变为小写，并写入文件
    out_f.close()


if __name__ == "__main__":
    """
    用于data_multi30k.sh中的第一句：python data_multi30k.py --pair_dir $1 --dest_dir $2 --src_lang $3 --trg_lang $4
    例如:
    parser.add_argument(
        "-p",
        "--pair_dir",
        default=None,
        type=str,
        help="The directory which contains language pair files.",
    )
    在sh脚本中执行# !sh data_multi30k_back.sh wmt16 wmt16_cut  de  en
    这里就会传入                              ^$1    ^$2       ^$3 ^$4
    等会使用parser.pair_dir就可以获得输入的值$1(wmt16)
    """
    parser = argparse.ArgumentParser() # 创建解析器
    parser.add_argument(
        "-p",
        "--pair_dir",
        default=None,
        type=str,
        help="The directory which contains language pair files.",
    )
    parser.add_argument(
        "-d",
        "--dest_dir",
        default=None,
        type=str,
        help="The destination directory to save processed train, dev and test file.",
    )
    parser.add_argument("--src_lang", default="de", type=str, help="source language")
    parser.add_argument("--trg_lang", default="en", type=str, help="target language")

    args = parser.parse_args() # 解析参数，args是一个列表，包含了传递的参数值
    if not args.pair_dir:#如果不传参，就抛异常
        raise ValueError("Please specify --pair_dir")
    #判断args.dest_dir是否存在,不存在就创建
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    local_data_path = Path(args.pair_dir) # 获取本地数据路径
    data_dir = Path(args.dest_dir) # 获取保存路径
    """——————————————————————————下面的分词的实际例子，以第一轮循环“train”为例——————————————————————————
    # 处理德语训练集
    moses_cut(
        local_data_path / "train.de",  # 读取: /data/de-en/train.de
        data_dir / "train_src.cut.txt",  # 写入: /data/processed/train_src.cut.txt
        lang="de",  # 使用德语分词器
    )
    # 输出: [train] 源语言文本分词完成

    # 处理英语训练集
    moses_cut(
        local_data_path / "train.en",  # 读取: /data/de-en/train.en
        data_dir / "train_trg.cut.txt",  # 写入: /data/processed/train_trg.cut.txt
        lang="en",  # 使用英语分词器
    )
    # 输出: [train] 目标语言文本分词完成
    """
    # 分词
    for mode in ["train", "val", "test"]:
        moses_cut(
            local_data_path / f"{mode}.{args.src_lang}", # 读取源语言文件
            data_dir / f"{mode}_src.cut.txt",
            lang=args.src_lang,
        )
        print(f"[{mode}] 源语言文本分词完成")
        moses_cut(
            local_data_path / f"{mode}.{args.trg_lang}", # 读取目标语言文件
            data_dir / f"{mode}_trg.cut.txt",
            lang=args.trg_lang,
        )
        print(f"[{mode}] 目标语言文本分词完成")
    # 创建文件夹，移动读取的文本到刚创建的文件夹里
    # if not data_dir.exists():
    #     data_dir.mkdir(parents=True)
    # for fpath in local_data_path.glob("*.txt"): # 遍历所有分词后的文件,并移动到目标文件夹
    #     fpath.rename(data_dir / fpath.name)
