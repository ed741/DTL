#include <iostream>
#include "taco.h"
#include "taco/format.h"

#define I ##I##
#define J ##J##

#define NNZ_A ##NNZ_A##
#define NNZ_B ##NNZ_B##
#define NNZ_C ##NNZ_C##

#define E ##Epsilon##

using namespace taco;

extern "C" void init_A(Tensor<float>** r_a, Tensor<float>** a) {
//    std::cout << "init_A" << std::endl;
    Format f_a(##A_Format##);
    Tensor<float>* n = new Tensor<float>("A", {I,J}, f_a);
    *r_a = n;
    *a = n;
}

extern "C" void dealloc_A(Tensor<float>** r_a_ptr) {
//    std::cout << "dealloc_A" << std::endl;
    Tensor<float>* r_a = *r_a_ptr;
    delete r_a;
}

extern "C" void init_B(Tensor<float>** r_b, Tensor<float>** b) {
//    std::cout << "init_B" << std::endl;
    Format f_b(##B_Format##);
    Tensor<float>* n = new Tensor<float>("B", {J}, f_b);
    *r_b = n;
    *b = n;
}

extern "C" void dealloc_B(Tensor<float>** r_b_ptr) {
//    std::cout << "dealloc_B" << std::endl;
    Tensor<float>* r_b = *r_b_ptr;
    delete r_b;
}

extern "C" void init_C(Tensor<float>** r_c, Tensor<float>** c) {
//    std::cout << "init_C" << std::endl;
    Format  f_c(##C_Format##);
    Tensor<float>* n = new Tensor<float>("C", {I}, f_c);
    *r_c = n;
    *c = n;
}

extern "C" void dealloc_C(Tensor<float>** r_c_ptr) {
//    std::cout << "dealloc_C" << std::endl;
    Tensor<float>* r_c = *r_c_ptr;
    delete r_c;
}

extern "C" void setup_A(Tensor<float>** a_ptr, int32_t* coord_0, int32_t* coord_1, float* vals){
//    std::cout << "# setup_A" << std::endl;
    Tensor<float>* a = *a_ptr;
    for(uint64_t idx = 0; idx < NNZ_A; idx++){

        int32_t c_0 = coord_0[idx];
        int32_t c_1 = coord_1[idx];
        float val = vals[idx];
        (*a)(c_0, c_1) = val;
    }
    a->pack();
}

extern "C" void setup_B(Tensor<float>** b_ptr, int32_t* coord_0, float* vals){
//    std::cout << "setup_B" << std::endl;
    Tensor<float>* b = * b_ptr;
    for(uint64_t idx = 0; idx < NNZ_B; idx++){
        int32_t c_0 = coord_0[idx];
        float val = vals[idx];
        (*b)(c_0) = val;
    }
    b->pack();
}

extern "C" void prepare(Tensor<float>** a_ptr, Tensor<float>** b_ptr, Tensor<float>** c_ptr) {
//    std::cout << "prepare" << std::endl;
    Tensor<float> a = ** a_ptr;
    Tensor<float> b = ** b_ptr;
    Tensor<float> c = ** c_ptr;
    IndexVar i, j, k;
    c(i) = a(i,j) * b(j);
    c.compile();
//    c.assemble();
    auto code = c.getSource();
    //Using stringstreams
	std::istringstream iss(code);
	std::string line;
	int idx = 0;
	while(std::getline(iss, line)) {
		std::cout << "# " << idx++ << ": " << line << std::endl;
	}
}

extern "C" void spmspv(Tensor<float>** c_ptr, Tensor<float>** a_ptr, Tensor<float>** b_ptr) {
//    std::cout << "matmul" << std::endl;
    Tensor<float> c = ** c_ptr;
    c.assemble();
    c.compute();
}

extern "C" void check_C(uint64_t* correct_ret, float* total_error_ret, uint64_t* consistent_ret, Tensor<float>** c_ptr, int32_t* coord_0, float* vals, Tensor<float>** c_ptr_f) {
//    std::cout << "check_c" << std::endl;
    Tensor<float> c = **c_ptr;
    Tensor<float> c_f = **c_ptr_f;
    bool total_correct = 1;
    float total_error = 0;
    bool total_consistent = 1;

    auto value = c.begin();
    auto value_f = c_f.begin();
    int32_t idx = 0;
    bool check_consistent = 1;
//    std::cout << "# NNZ: " << NNZ_C << std::endl;
    while((value != c.end()) && (idx < NNZ_C)) {
//        std::cout << "# idx: " << idx << std::endl;
        auto coord = value->first;
        float c_v = value->second;
        int c_i = coord[0];
//        std::cout << "# c_i: " << c_i << std::endl;

        int32_t ref_i = coord_0[idx];
        float ref_v = vals[idx];
//        std::cout << "# ref_i: " << ref_i << std::endl;

//        std::cout << "# ref_v: " << ref_v << std::endl;
//        std::cout << "# c_v: " << c_v << std::endl;
        if (ref_i > c_i) {
            ref_v = 0.0;
        }
        if (c_v > ref_i) {
            c_v = 0.0;
        }
//        std::cout << "# ref_v: " << ref_v << " # c_v: " << c_v << std::endl;

        float error = (c_v - ref_v);
//        std::cout << "# error: " << error << std::endl;
        error = error > -error ? error : -error;
        float base_epsilon = E;
        float epsilon = base_epsilon * (ref_v > -ref_v? ref_v : -ref_v);
        bool correct = epsilon >= error;

//        std::cout << "# check_consistent: " << check_consistent << std::endl;
        if (check_consistent){
            bool coord_check = coord == value_f->first;
            float c_f_v = value_f->second;
            if (!coord_check) {
                    std::cout << "# coord mismatch!" << std::endl;
                    std::cout << "# coord: " << coord << " f coord: " << value_f->first << std::endl;
            }
            bool consistent = c_f_v == c_v;
            consistent &= coord_check;
            total_consistent &= consistent;
        }

        total_correct &= correct;
        total_error += error;


        if (ref_i > c_i) {
            ++value;
            ++value_f;
            check_consistent = 1;
        }
        if (c_v > ref_i) {
            ++idx;
            check_consistent = 0;
        }
        if (c_i == ref_i) {
            ++value;
            ++value_f;
            ++idx;
            check_consistent = 1;
        }

    }
    *correct_ret = (uint64_t) total_correct;
    *total_error_ret = total_error;
    *consistent_ret = (uint64_t) total_consistent;
//    std::cout << "*correct:" << *_correct << " *total_error:" << *_total_error << " *consistent:" << *_consistent << std::endl;
}
