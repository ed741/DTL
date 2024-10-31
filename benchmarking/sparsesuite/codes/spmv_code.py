from benchmarking.benchmark import TestCode

spmv_base_code = TestCode(
    setup="""
assert False
""",
    benchmark="""
lib.spmv(c, a, b)
""",
    test="""
results = lib.check_C(c, ref_c, f_c)
correct = bool(results[0].value)
total_error = float(results[1].value)
consistent = bool(results[2].value)
""",
    clean="""
assert False
""",
)

spmv_triple_code = TestCode(
    setup="""
root, (a,b,c) = lib.init()
lib.setup_A(a, ref_a_0, ref_a_1, ref_a_val)
lib.setup_B(b, ref_b)
lib.prepare(a,b,c)
""",
    benchmark=spmv_base_code.benchmark,
    test=spmv_base_code.test,
    clean="""
lib.dealloc(root)
""",
)

spmv_pair_code = TestCode(
    setup="""
r_ab, (a, b) = lib.init_AB()
r_c, (c) = lib.init_C()
lib.setup_A(a, ref_a_0, ref_a_1, ref_a_val)
lib.setup_B(b, ref_b)
lib.prepare(a,b,c)
""",
    benchmark=spmv_base_code.benchmark,
    test=spmv_base_code.test,
    clean="""
lib.dealloc_AB(r_ab)
lib.dealloc_C(r_c)
""",
)

spmv_single_code = TestCode(
    setup="""
r_a, (a) = lib.init_A()
r_b, (b) = lib.init_B()
r_c, (c) = lib.init_C()
lib.setup_A(a, ref_a_0, ref_a_1, ref_a_val)
lib.setup_B(b, ref_b)
lib.prepare(a,b,c)
""",
    benchmark=spmv_base_code.benchmark,
    test=spmv_base_code.test,
    clean="""
lib.dealloc_A(r_a)
lib.dealloc_B(r_b)
lib.dealloc_C(r_c)
""",
)