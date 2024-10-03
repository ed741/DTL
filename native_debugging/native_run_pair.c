#include <stdio.h>
#include <stdint.h>

typedef struct {
    void* data;
} ABPtrR;

typedef struct {
    void* data;
} APtr;

typedef struct {
    void* data;
} BPtr;

typedef struct {
    void* data;
} CPtrR;

typedef struct {
    void* data;
} CPtr;

extern void init_AB(ABPtrR* r_ab, APtr* a, BPtr* b);
extern void init_C(CPtrR* r_c, CPtr* c);
extern void dealloc_AB(ABPtrR* r_ab);
extern void dealloc_C(CPtrR* r_c);


//
//extern void setup_A(APtr* a, float* array);
//extern void setup_B(BPtr* b, float* array);
extern void set_A(APtr* a, uint64_t d1, uint64_t d2, float array);
extern void set_B(BPtr* b, uint64_t d1, uint64_t d2, float array);

extern void matmul(CPtr* c, APtr* a, BPtr* b);

int main() {
   // printf() displays the string inside quotation
   printf("starting\n");
   printf("init_AB?\n");
   ABPtrR r_ab;
   APtr a;
   BPtr b;
   init_AB(&r_ab, &a, &b);
   printf("done:  root: %p  a: %p  b:%p\n", r_ab.data, a.data, b.data);
   printf("init_C?\n");
   CPtrR r_c;
   CPtr c;
   init_C(&r_c, &c);
   printf("done:  root: %p  c: %p\n", r_c.data, c.data);

   printf("setup_A?\n");
   for(int i = 0; i < 8; i++){
       for(int j = 0; j < 8; j++){
           float val = 0.1;
           set_A(&a, i, j, val);
      }
   }
//   float a_array[64];
//   for(int i = 0; i < 64; i++){
//       a_array[i] = 0.1;
//   }
//   setup_A(&a, a_array);
   printf("done\n");

   printf("setup_B?\n");
   for(int i = 0; i < 8; i++){
       for(int j = 0; j < 8; j++){
           float val = -0.1;
           set_B(&b, i, j, val);
      }
   }
//   float b_array[64];
//   for(int i = 0; i < 64; i++){
//    b_array[i] = 0.1;
//   }
//   setup_B(&b, b_array);
   printf("done\n");


   printf("matmul?\n");
   matmul(&c, &a, &b);
   printf("done\n");



   printf("dealloc AB\n");
   dealloc_AB(&r_ab);
   printf("done\n");
   printf("dealloc C\n");
   dealloc_C(&r_c);
   printf("done\n");


   return 0;
}