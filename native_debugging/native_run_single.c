#include <stdio.h>

typedef struct {
    void* data;
} APtr;

typedef struct {
    void* data;
} BPtr;

typedef struct {
    void* data;
} CPtr;

extern void init_A(APtr* r_a, APtr* a);
extern void init_B(BPtr* r_a, BPtr* a);
extern void init_C(CPtr* r_a, CPtr* a);

int main() {
   // printf() displays the string inside quotation
   printf("starting\n");
   printf("init_A?\n");
   APtr r_a;
   APtr a;
   init_A(&r_a, &a);
   printf("done: %p %p\n", r_a.data, a.data);
   printf("init_B?\n");
   BPtr r_b;
   BPtr b;
   init_B(&r_b, &b);
   printf("done: %p %p\n", r_b.data, b.data);
   printf("init_C?\n");
   CPtr r_c;
   CPtr c;
   init_C(&r_c, &c);
   printf("done: %p %p\n", r_c.data, c.data);

   return 0;
}
