from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("tree", sources=["tree.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=[], language="c++",
        ),
    Extension("qt_tree", sources=["qt_tree.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[],
        extra_compile_args=[], language="c++",
        ),
]
setup(
    ext_modules=cythonize(extensions),
)
