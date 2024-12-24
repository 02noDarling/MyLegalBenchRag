#!/bin/bash

# 安装 pip-tools，使用清华镜像源
pip install pip-tools --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 同步依赖并安装当前包（开发模式），使用清华镜像源
pip-sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple
