import os
import subprocess
import tempfile
from io import StringIO

from xdsl.dialects import builtin
from xdsl.printer import Printer


def mlir_compile(module: builtin.ModuleOp, lib_output: str, llvm_out: str = None, verbose = 2):
    # if header_out==None:
    #     header_out = lib_output.removesuffix(".o") + ".h"

    if verbose > 0:
        print(f"Compile to Binary: {lib_output}")

    if verbose > 1:
        print("Module:")
        print(module)

    if verbose > 1:
        print("mlir output:")
    res = StringIO()
    printer = Printer(print_generic_format=False, stream=res)
    printer.print(module)
    if verbose > 1:
        print(res.getvalue())

    fd, path = tempfile.mkstemp()
    if verbose > 0:
        print(f"Making tmp mlir - IR file: {path}")
    with os.fdopen(fd, 'wb') as tmp:
        tmp.write(res.getvalue().encode('utf8'))


    if verbose > 1:
        print("mlir-opt:")
    passes = [
        "--convert-math-to-funcs",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-func-to-llvm",
        "--convert-arith-to-llvm",
        "--expand-strided-metadata",
        "--normalize-memrefs",
        "--memref-expand",
        "--fold-memref-alias-ops",
        "--finalize-memref-to-llvm",
        "--reconcile-unrealized-casts",
    ]
    if verbose > 1:
        print("command:")
        print(' '.join(['mlir-opt'] + passes))
    process_opt = subprocess.Popen(['mlir-opt'] + passes, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = process_opt.communicate(res.getvalue().encode('utf8'))
    process_opt.wait()
    if verbose > 1:
        print("stdout:")
        print(out.decode('utf8'))
        print("stderr:")
        print(err.decode('utf8') if err is not None else None)

    if verbose > 1:
        print("mlir-translate")
    process_translate = subprocess.Popen(['mlir-translate', '--mlir-to-llvmir'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = process_translate.communicate(out)
    process_translate.wait()
    if verbose > 1:
        print("stdout:")
        print(out.decode('utf8'))
        print("stderr:")
        print(err.decode('utf8') if err is not None else None)

    if llvm_out is None:
        fd, path = tempfile.mkstemp(suffix=".ll")
    else:
        fd = open(llvm_out, "wb")
        path = llvm_out

    if verbose > 0:
        print(f"Making tmp llvm-IR file: {path}")
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(out)
            tmp.flush()

        clang_compile(path, lib_output, verbose)
    finally:
        # os.remove(path)
        pass

    if verbose > 0:
        print("Done compiling with mlir / clang")


def clang_compile(llvm_path: str, lib_output: str, verbose: int = 2):
    clang_args = ["clang"]
    clang_args.extend(['-o', lib_output])
    clang_args.append('-shared')
    # clang_args.append("-c")
    # clang_args.append("-v")
    clang_args.append("-g")
    clang_args.append("-O3")
    clang_args.append(llvm_path)

    if verbose > 1:
        print(" ".join(clang_args))
    process_clang = subprocess.Popen(clang_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    # out, err = process_clang.communicate(out)
    # print("stdout:")
    # print(out.decode('utf8') if out is not None else None)
    # print("stderr:")
    # print(err.decode('utf8') if err is not None else None)
    process_clang.wait()

