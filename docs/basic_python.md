
# Python 基础知识

### 路径相关

在 Python 中有两种路径, 分别是:

+ [系统路径](https://docs.python.org/3/library/sys.html#sys.path)
+ [工作路径](https://docs.python.org/3/library/os.html#os.chdir)

Python 会从系统路径中找模块, 以工作路径为基准将相对路径转换为绝对路径。

如果运行在 `/home/lqxu/IntroNLP/` 目录夹下运行 `python ./examples/sentence_embedding/01_u_sim_cse.py` 指令, 
此时 Python 解释器做对于路径的设置如下:

+ 会将运行文件所在的目录添加入 **系统路径**, 也就是 `/home/lqxu/IntroNLP/examples/sentence_embedding/` 路径
+ 会将当前目录作为 **工作路径**, 也就是 `/home/lqxu/IntroNLP/`

此时在 `./` 目录夹下的模块文件都找不到, 解决办法是: `sys.path.insert(0, "./")`

在 PyCharm 中, 会自动将文件所在的目录作为工作路径, 并将项目路径添加到系统路径中。如有需要, 可以更改 Python 运行的默认模板。

无论哪一种方式都很坑 !!! 请确保正确设置路径。

### Python 字符串
