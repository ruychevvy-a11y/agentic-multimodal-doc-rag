from docrag import config
from docrag.dataset import mmlongbench, mpdocvqa

# 数据集名 --> 造这个数据集的函数；标注格式各不相同，一个数据集一个模块
BUILDERS = {
    "mpdocvqa": mpdocvqa.build,
    "mmlongbench": mmlongbench.build,
}

if __name__ == "__main__":
    BUILDERS[config.DATASET]()
