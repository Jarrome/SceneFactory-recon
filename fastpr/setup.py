'''
    Based on https://github.com/wuminye/PCPR

    Yijun Yuan
    Coded at: Feb. 1. 2024
'''

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fastpr',
    version="0.5",
    ext_modules=[
        CUDAExtension('fastpr', [
            'src/pr.cpp',
            'src/raster.cu'

        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
