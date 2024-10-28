#include <iostream>
#include "taco.h"
#include "taco/format.h"

#define I ##I##
#define J ##J##

#define NNZ ##NNZ##

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
    for(uint64_t idx = 0; idx < NNZ; idx++){

        int32_t c_0 = coord_0[idx];
        int32_t c_1 = coord_1[idx];
        float val = vals[idx];
        (*a)(c_0, c_1) = val;
    }
    a->pack();
}

extern "C" void setup_B(Tensor<float>** b_ptr, float* vals){
//    std::cout << "setup_B" << std::endl;
    Tensor<float>* b = * b_ptr;
    for(int j = 0; j < J; j++){
            (*b)(j) = vals[j];
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
}

extern "C" void spmv(Tensor<float>** c_ptr, Tensor<float>** a_ptr, Tensor<float>** b_ptr) {
//    std::cout << "matmul" << std::endl;
    Tensor<float> c = ** c_ptr;
    c.assemble();
    c.compute();
}

extern "C" void check_C(uint64_t* correct_ret, float* total_error_ret, uint64_t* consistent_ret, Tensor<float>** c_ptr, float* vals, Tensor<float>** c_ptr_f) {
//    std::cout << "check_c" << std::endl;
    Tensor<float> c = **c_ptr;
    Tensor<float> c_f = **c_ptr_f;
    bool total_correct = 1;
    float total_error = 0;
    bool total_consistent = 1;

    auto value_f = c_f.begin();
    for(auto value = c.begin(); value != c.end(); ++value) {
        auto coord = value->first;
        float c_v = value->second;
//        std::cout << "coord: " << coord << " val: " << c_v << std::endl;
        int i = coord[0];
//        std::cout << "i: " << i << std::endl;

        float ref = vals[i];
        float error = (c_v - ref);
        error = error > -error ? error : -error;
        float base_epsilon = E;
        float epsilon = base_epsilon * (ref > -ref? ref : -ref);
        bool correct = epsilon >= error;

        bool coord_check = value->first == value_f->first;
        float c_f_v = value_f->second;
        if (!coord_check) {
                std::cout << "coord mismatch!" << std::endl;
                std::cout << "coord: " << coord << " f coord: " << value_f->first << std::endl;
        }
        bool consistent = c_f_v == c_v;
        consistent &= coord_check;

        total_correct &= correct;
        total_error += error;
        total_consistent &= consistent;
        ++value_f;
    }
    *correct_ret = (uint64_t) total_correct;
    *total_error_ret = total_error;
    *consistent_ret = (uint64_t) total_consistent;
//    std::cout << "*correct:" << *_correct << " *total_error:" << *_total_error << " *consistent:" << *_consistent << std::endl;
}
