import glob
from pathlib import Path
import airspeed
from setuptools import setup, find_packages
from distutils.command.clean import clean
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.cuda import is_available as torch_cuda_available
from activations import __version__
import os
# degrees = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (5, 4), (7, 6)]
degrees = [(5, 4), (7, 6)]
name = 'activation-functions'

#import ipdb; ipdb.set_trace()
# find_packages(where="rational")


def is_torch_cuda_available():
    """Wrapper for torch cuda availability check (torch.cuda.is_available) that takes an environment variable
    'FORCE_CUDA' into account and returns also true iff FORCE_CUDA=1.

    This is necessary when building rational in a Dockerfile script since the docker build pass doesn't have
    access to cuda and thus torch.cuda.is_available always returns false, even when the docker image which is
    to be built in fact does have cuda.
    """
    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    return force_cuda or torch_cuda_available()


def generate_cpp_module(fname, degrees=degrees, versions=None):
    file_content = airspeed.Template("""
\#include <torch/extension.h>
\#include <vector>
\#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


#foreach ($vname in $versions)
#if( $vname == 'D' )
#set ($forward_header = 'const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d')
#set ($backward_header = 'const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d')
#set ($forward_invocation = 'training, iteration, x, n, d')
#set ($backward_invocation = 'training, iteration, grad_output, x, n, d')
#else
#set ($forward_header = 'torch::Tensor x, torch::Tensor n, torch::Tensor d')
#set ($backward_header = 'torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d')
#set ($forward_invocation = 'x, n, d')
#set ($backward_invocation = 'grad_output, x, n, d')
#end
    #foreach ($degs in $degrees)
	at::Tensor rational_cuda_forward_${vname}_$degs[0]_$degs[1]($forward_header);
    std::vector<torch::Tensor> rational_cuda_backward_${vname}_$degs[0]_$degs[1]($backward_header);
    #end


    #foreach ($degs in $degrees)
    at::Tensor rational_forward_${vname}_$degs[0]_$degs[1]($forward_header) {
        CHECK_INPUT(d);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        return rational_cuda_forward_${vname}_$degs[0]_$degs[1]($forward_invocation);
    }
    std::vector<torch::Tensor> rational_backward_${vname}_$degs[0]_$degs[1]($backward_header) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_${vname}_$degs[0]_$degs[1]($backward_invocation);
    }
    #end
#end

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#foreach ($degs in $degrees)
    #foreach ($vname in $versions)
    m.def("forward_${vname}_$degs[0]_$degs[1]", &rational_forward_${vname}_$degs[0]_$degs[1], "Rational forward ${vname}_$degs[0]_$degs[1]");
    m.def("backward_${vname}_$degs[0]_$degs[1]", &rational_backward_${vname}_$degs[0]_$degs[1], "Rational backward ${vname}_$degs[0]_$degs[1]");
    #end
#end
}
    """)
    content = file_content.merge(locals())

    with open(fname, "w") as text_file:
        text_file.write(content)


def generate_cpp_kernels_module(fname, degrees=degrees, template_contents=None):
    degrees = [[e[0], e[1], max(e[0], e[1])] for e in degrees]

    template = """
\#include <torch/extension.h>
\#include <ATen/cuda/CUDAContext.h>
\#include <cuda.h>
\#include <cuda_runtime.h>
\#include <vector>
\#include <stdlib.h>

\#include <curand.h>
\#include <curand_kernel.h>
\#include <curand_philox4x32_x.h>

constexpr uint32_t THREADS_PER_BLOCK = 512;
"""

    file_content = airspeed.Template(template + template_contents)

    content = file_content.merge(locals())

    with open(fname, "w") as text_file:
        text_file.write(content)


if is_torch_cuda_available():
    version_names = []
    template_contents = ""
    for template_fname in sorted(glob.glob("activations/torch/rationals/_cuda/versions/*.cu")):
        version_names.append(Path(template_fname).stem)
        with open(template_fname) as infile:
            template_contents += infile.read()

    generate_cpp_module(
        fname='activations/torch/rationals/_cuda/rational_cuda.cpp', versions=version_names)
    generate_cpp_kernels_module(
        fname='activations/torch/rationals/_cuda/rational_cuda_kernels.cu', template_contents=template_contents)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.readlines()


class clean_all(clean):
    def run(self):
        self.all = True
        super().run()
        import shutil
        import os
        egginf = name.replace('-', '_')
        shutil.rmtree(egginf + '.egg-info')
        shutil.rmtree('dist')
        if os.path.exists("activations/torch/rationals/cuda.cpython-36m-x86_64-linux-gnu.so"):
            os.remove(
                "activations/torch/rationals/cuda.cpython-36m-x86_64-linux-gnu.so")
        print("Cleaned everything")


setup(
    name=name,
    version=__version__,
    author="Quentin Delfosse, Patrick Schramowski",
    author_email="quentin.delfosse@cs.tu-darmstadt.de",
    description="Activations functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/k4ntz/activation_functions",
    packages=find_packages(exclude=["tests"]),
    package_data={'': ['*.json']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License"
    ],
    install_requires=requirements,
    ext_modules=[
        CUDAExtension('activations.torch.rationals.cuda', [
            'activations/torch/rationals/_cuda/rational_cuda.cpp',
            'activations/torch/rationals/_cuda/rational_cuda_kernels.cu',
        ],
            extra_compile_args={'cxx': [],
                                'nvcc': ['-gencode=arch=compute_80,code="sm_80,compute_80"', '-lineinfo']
                                }
        ),
    ] if is_torch_cuda_available() else [],
    cmdclass={
        'build_ext': BuildExtension,
        'clean': clean_all
    },
    setup_requires=['airspeed'],
    python_requires='>=3.5.0',)
