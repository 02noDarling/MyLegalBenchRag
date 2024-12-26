from setuptools import setup, find_packages

setup(
    name='MyLegalBenchRag',
    version='0.1',
    package_dir={'': 'src'},  # 指定源码目录为 src
    packages=find_packages(where='src'),  # 在 src 目录下查找包
    python_requires='>=3.12',  # 支持的 Python 版本
)
