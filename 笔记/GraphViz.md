最近在画基于keras的模型可视化，遇到这个错误，查阅多篇论文最后完美解决。现总结一下解决过程。

首先下载下面三个模块

pip install graphviz

pip install pydot

pip install pydot_ng

下载这三个还不能解决这个问题，还需要安装GraphViz

http://www.graphviz.org/

在win10上直接运行安装。安装路径都可以默认。

设置环境变量
首先，按win+e键弹出文件窗口
然后，右键此电脑 →属性→高级系统设置→环境变量，然后点击系统变量列表中的Path，点击编辑就可以 ，选出Graphviz2.38/bin的路径 。

然后

>>>import os

>>>os.environ.get('PATH', '')

看编辑的路劲是否在列表中，如果在就说明成功了。

运行程序，还出现错误，就加上下面的程序

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
在运行程序，还是失败，就去找到pydot的源码，在pydot.py中找到类Dot，修改self.prog = 'dot'为self.prog = 'dot.exe'，之后测试，成功运行程序。

借鉴

https://blog.csdn.net/leviopku/article/details/81433867

https://blog.csdn.net/sinat_38653840/article/details/84776806
————————————————
版权声明：本文为CSDN博主「富哥92」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/fuge92/article/details/88371693