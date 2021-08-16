#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#define N 29
#define BPG 2
//Declare functions
void Allocate_Memory();
void Free_Memory();
void friction();
void jaco(float *x , float *r);
void comR(float *d_r, float *x);
void UpdateQ(float *x ,float*dx);
void Send_To_Device();
void Get_From_Device();
void Call_GPUSerial_Function();
void Init();
