"""Build stub for the optional Linux-only GDN CUDA extension (OpenSpec 4.1).

Targets Volta ``sm_70`` for the college V100S host under the pinned cu126
toolchain. This stub only declares the extension layout; the real fused
forward/backward sources land once the pilot/toolchain is confirmed.

Linux-only: do not run on Windows/CPU checkouts. CPU smoke tests must never
require this file, a CUDA toolkit, or ``nvcc``.

Usage (Linux V100S host only)::

    python setup_gdn_cuda.py build_ext --inplace
"""

from __future__ import annotations

import platform

LINUX_ONLY = True
TARGET_ARCH = "sm_70"
NVCC_GENCODE = ["-gencode", "arch=compute_70,code=sm_70"]
SOURCES = ["gdn_cuda/csrc/bindings.cpp", "gdn_cuda/csrc/gdn_fused.cu"]


def get_extension_kwargs() -> dict:
    return {
        "name": "gdn_cuda._fused",
        "sources": list(SOURCES),
        "extra_compile_args": {
            "cxx": ["-O3"],
            "nvcc": ["-O3", *NVCC_GENCODE, "--use_fast_math"],
        },
    }


def build() -> None:
    if platform.system() != "Linux":
        raise RuntimeError(
            f"GDN CUDA extension build is Linux-only (running on {platform.system()}). "
            "Build on the V100S host with the cu126 toolchain."
        )
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError as exc:
        raise RuntimeError(
            "Cannot build the GDN CUDA extension: torch with CUDA support is required "
            f"on the build host ({exc})."
        ) from exc
    from setuptools import setup

    setup(
        name="gdn-cuda-fused",
        version="0.0.0-scaffold",
        ext_modules=[CUDAExtension(**get_extension_kwargs())],
        cmdclass={"build_ext": BuildExtension},
    )


if __name__ == "__main__":
    build()
