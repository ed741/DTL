#include <iostream>
#include <sstream>
#include "taco.h"
#include "taco/format.h"

#define I ##I##
#define J ##J##
#define K ##K##

#define E ##Epsilon##

using namespace taco;

extern "C" void init_A(Tensor<float>** r_a, Tensor<float>** a) {
//    std::cout << "init_A" << std::endl;
    Format f_a(##A_Format##);
    Tensor<float>* n = new Tensor<float>("A", {I,J}, f_a);
    *r_a = n;
    *a = n;
    n->pack();
}

extern "C" void dealloc_A(Tensor<float>** r_a_ptr) {
//    std::cout << "dealloc_A" << std::endl;
    Tensor<float>* r_a = *r_a_ptr;
    delete r_a;
}

extern "C" void init_B(Tensor<float>** r_b, Tensor<float>** b) {
//    std::cout << "init_B" << std::endl;
    Format f_b(##B_Format##);
    Tensor<float>* n = new Tensor<float>("B", {J,K}, f_b);
    *r_b = n;
    *b = n;
    n->pack();
}

extern "C" void dealloc_B(Tensor<float>** r_b_ptr) {
//    std::cout << "dealloc_B" << std::endl;
    Tensor<float>* r_b = *r_b_ptr;
    delete r_b;
}

extern "C" void init_C(Tensor<float>** r_c, Tensor<float>** c) {
//    std::cout << "init_C" << std::endl;
    Format  f_c(##C_Format##);
    Tensor<float>* n = new Tensor<float>("C", {I,K}, f_c);
    *r_c = n;
    *c = n;
    n->pack();
}

extern "C" void dealloc_C(Tensor<float>** r_c_ptr) {
//    std::cout << "dealloc_C" << std::endl;
    Tensor<float>* r_c = *r_c_ptr;
    delete r_c;
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

extern "C" void setup_B(Tensor<float>** b_ptr, float* vals){
//    std::cout << "setup_B" << std::endl;
    Tensor<float>* b = * b_ptr;
    for(int j = 0; j < J; j++){
        for(int k = 0; k < K; k++){
            (*b)(j, k) = vals[K*j + k];
        }
    }
    b->pack();
}

extern "C" void prepare(Tensor<float>** a_ptr, Tensor<float>** b_ptr, Tensor<float>** c_ptr) {
//    std::cout << "prepare" << std::endl;
    Tensor<float> a = ** a_ptr;
    Tensor<float> b = ** b_ptr;
    Tensor<float> c = ** c_ptr;
    IndexVar i, j, k;
    c(i,k) = a(i,j) * b(j,k);
    c.compile();
//    c.assemble();
//    auto code = c.getSource();
//    //Using stringstreams
//	std::istringstream iss(code);
//	std::string line;
//	int idx = 0;
//	while(std::getline(iss, line)) {
//		std::cout << "# " << idx++ << ": " << line << std::endl;
//	}
}

extern "C" void matmul(Tensor<float>** c_ptr, Tensor<float>** a_ptr, Tensor<float>** b_ptr) {
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

    bool checked[I*K];

    auto value_f = c_f.begin();
    for(auto value = c.begin(); value != c.end(); ++value) {
        auto coord = value->first;
        float c_v = value->second;
//        std::cout << "# coord: " << coord << " val: " << c_v << std::endl;
        int i = coord[0];
        int k = coord[1];
//        std::cout << "# i: " << i << " k: " << k << std::endl;
        checked[K*i + k] = 1;

        float ref = vals[K*i + k];
        float error = (c_v - ref);
        error = error > -error ? error : -error;
        float base_epsilon = E;
        float epsilon = base_epsilon * (ref > -ref? ref : -ref);
        bool correct = epsilon >= error;
        if (!correct) {
            std::cout << "# result incorrect! at i: " << i << ", k: " << k << " :: result: " << c_v << " ref: " << ref << std::endl;
//            std::cout << std::format("# result incorrect! at i: {}, k: {} :: result: {}, ref: {}\n", i, k, c_v, ref);
        }
        bool coord_check = value->first == value_f->first;
        float c_f_v = value_f->second;
        if (!coord_check) {
                std::cout << "# coord mismatch!" << std::endl;
//                std::cout << std::format("# coord mismatch!: {},  f coord: {}\n", coord, value_f->first);
        }
        bool consistent = c_f_v == c_v;
        consistent &= coord_check;

        total_correct &= correct;
        total_error += error;
        total_consistent &= consistent;
        ++value_f;
    }

    bool all_check = true;
    for(int i = 0; i < I; i++){
        for(int k = 0; k < K; k++){
            float ref = vals[K*i + k];
            if (ref != 0.0f) {
                bool c = checked[K*i + k];
                if (!c) {
                    std::cout << "# result at i: " << i << ", k: " << k << " is " << ref << " but was not iterated in C. " << std::endl;
//                    std::cout << std::format("# result at i: {}, k: {} is {} but was not iterated in C. \n", i, k, ref);
                }
                all_check &= c;
            }
        }
    }
    total_correct &= all_check;

//    for(int i = 0; i < I; i++){
//        for(int k = 0; k < K; k++){
//            float c_v = c.at({i, k});
////            float c_v = 0;
//            float ref = vals[K*i + k];
//            float error = (c_v - ref);
//            error = error > -error ? error : -error;
//            float base_epsilon = E;
//            float epsilon = base_epsilon * (ref > -ref? ref : -ref);
//            bool correct = epsilon >= error;
//            float c_f_v = c_f.at({i, k});
////            float c_f_v = 0;
//            bool consistent = c_f_v == c_v;
//
//            total_correct &= correct;
//            total_error += error;
//            total_consistent &= consistent;
//
////            std::cout << "i:" << i << " k:"<< k << " c[i,k]:" << c_v << " ref:" << ref << " error:" << error << " epsilon:" << epsilon << " correct:" << correct << " c_first:" << c_f_v << " consistent:" << consistent << " Total Error:" << total_error << std::endl;
//        }
////              std::cout << "i:" << i << " k:"<< "0-K" << " consistent:" << total_consistent << " Total Error:" << total_error << std::endl;
//    }
    *correct_ret = (uint64_t) total_correct;
    *total_error_ret = total_error;
    *consistent_ret = (uint64_t) total_consistent;
//    std::cout << "*correct:" << *_correct << " *total_error:" << *_total_error << " *consistent:" << *_consistent << std::endl;
}
