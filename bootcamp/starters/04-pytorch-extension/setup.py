from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory containing this file
this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='fused_layernorm_cuda',
    version='0.1.0',
    description='Fused LayerNorm CUDA Extension',
    author='CUDA Bootcamp',
    ext_modules=[
        CUDAExtension(
            name='fused_layernorm_cuda',
            sources=[
                os.path.join(this_dir, 'csrc/layernorm.cpp'),
                os.path.join(this_dir, 'csrc/layernorm_cuda.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-lineinfo',
                    '--use_fast_math',
                    '-std=c++17',
                    # Support multiple GPU architectures
                    '-gencode=arch=compute_70,code=sm_70',  # V100
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
                    '-gencode=arch=compute_90,code=sm_90',  # H100
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    packages=['fused_layernorm'],
    package_dir={'fused_layernorm': 'fused_layernorm'},
    python_requires='>=3.8',
    install_requires=['torch>=2.0.0'],
)
