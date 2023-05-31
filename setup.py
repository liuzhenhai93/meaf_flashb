from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='custom_attention',
    ext_modules=CUDAExtension(
        sources=['meaf_flashb.cc']
    )
)