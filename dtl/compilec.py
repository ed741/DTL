import os
import subprocess
import tempfile
from io import StringIO

from xdsl.dialects import builtin
from xdsl.printer import Printer


def mlir_compile(module: builtin.ModuleOp, lib_output: str, llvm_out: str = None, llvm_only: bool = False, verbose = 2):
    # if header_out==None:
    #     header_out = lib_output.removesuffix(".o") + ".h"

    if verbose > 0:
        print(f"Compile to Binary: {lib_output}")

    if verbose > 1:
        print("Module:")
        print(module)

    if verbose > 1:
        print("mlir output:")
    xdsl_module = StringIO()
    printer = Printer(print_generic_format=False, stream=xdsl_module)
    printer.print(module)
    if verbose > 1:
        print(xdsl_module.getvalue())


    # mlir_tmp_fd, mlir_tmp_path = tempfile.mkstemp()
    # if verbose > 0:
    #     print(f"Making tmp mlir - IR file: {mlir_tmp_path}")
    # with os.fdopen(mlir_tmp_fd, 'wb') as mlir_tmp:
    #     mlir_tmp.write(xdsl_module.getvalue().encode('utf8'))


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
    mlir_opt_out, err = process_opt.communicate(xdsl_module.getvalue().encode('utf8'))
    process_opt.wait()
    if verbose > 1:
        print("stdout:")
        print(mlir_opt_out.decode('utf8'))
        print("stderr:")
        print(err.decode('utf8') if err is not None else None)

    if verbose > 1:
        print("mlir-translate")
    process_translate = subprocess.Popen(['mlir-translate', '--mlir-to-llvmir'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    mlir_translate_out, err = process_translate.communicate(mlir_opt_out)
    process_translate.wait()
    if verbose > 1:
        print("stdout:")
        print(mlir_translate_out.decode('utf8'))
        print("stderr:")
        print(err.decode('utf8') if err is not None else None)

    if llvm_out is None:
        llvm_tmp_fd, llvm_out = tempfile.mkstemp(suffix=".ll")
        if verbose > 0:
            print(f"Making tmp llvm-IR file: {llvm_out}")
        try:
            with os.fdopen(llvm_tmp_fd, 'wb') as llvm_tmp:
                llvm_tmp.write(mlir_translate_out)
                llvm_tmp.flush()
            if not llvm_only:
                clang_compile(llvm_out, lib_output, verbose)
        finally:
            os.remove(llvm_out)
    else:
        if verbose > 0:
            print(f"Making llvm-IR file: {llvm_out}")
        os.makedirs(os.path.dirname(llvm_out), exist_ok=True)
        with open(llvm_out, "wb") as llvm_fd:
            llvm_fd.write(mlir_translate_out)
            llvm_fd.flush()
        if not llvm_only:
            clang_compile(llvm_out, lib_output, verbose)

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

