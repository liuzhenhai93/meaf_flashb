from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='meaf_flashb',
    ext_modules=CUDAExtension(
        sources=['meaf_flashb.cc']
    )
)