#include <iostream>
#include "taco.h"
#include "taco/format.h"

#define I ##I##
#define J ##J##

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

extern "C" void setup_A(Tensor<float>** a_ptr, float* vals){
//    std::cout << "setup_A" << std::endl;
    Tensor<float>* a = *a_ptr;
    for(int i = 0; i < I; i++){
        for(int j = 0; j < J; j++){
            (*a)(i, j) = vals[J*i + j];
        }
    }
    a->pack();
}

extern "C" void prepare(Tensor<float>** a_ptr) {
//    std::cout << "prepare" << std::endl;
}

extern "C" void func(Tensor<float>** a_ptr) {
//    std::cout << "func" << std::endl;
}

extern "C" void check_A(uint64_t* correct_ret, float* total_error_ret, uint64_t* consistent_ret, Tensor<float>** a_ptr, float* vals, Tensor<float>** a_ptr_f) {
//    std::cout << "check_a" << std::endl;
    Tensor<float> a = **a_ptr;
    Tensor<float> a_f = **a_ptr_f;
    bool total_correct = 1;
    float total_error = 0;
    bool total_consistent = 1;

    auto value_f = a_f.begin();
    for(auto value = a.begin(); value != a.end(); ++value) {
        auto coord = value->first;
        float a_v = value->second;
//        std::cout << "coord: " << coord << " val: " << c_v << std::endl;
        int i = coord[0];
        int j = coord[1];
//        std::cout << "i: " << i << " k: " << k << std::endl;

        float ref = vals[J*i + j];
        float error = (a_v - ref);
        error = error > -error ? error : -error;
        float base_epsilon = E;
        float epsilon = base_epsilon * (ref > -ref? ref : -ref);
        bool correct = epsilon >= error;

        bool coord_check = value->first == value_f->first;
        float a_f_v = value_f->second;
        if (!coord_check) {
                std::cout << "coord mismatch!" << std::endl;
                std::cout << "coord: " << coord << " f coord: " << value_f->first << std::endl;
        }
        bool consistent = a_f_v == a_v;
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
