#include"gpumain.h"
#include<time.h>

// Host variable
extern float *h_A;
extern float *h_x;
extern float *h_B;
extern float *h_R;
extern float *h_scalar;
extern float *Q;
extern float *R;
extern float *h_r;

// Device variable
extern float *d_A;
extern float *d_B;
extern float *d_x;
extern float *d_R;
extern float *d_p;
extern float *d_Ap;
extern float *d_temp;
extern float *d_scalar;

int main()
{

	// Intiailize clock and get the time
	clock_t t ; 
	t = clock();

	// Allocate memory in both device and host
	Allocate_Memory();
	// assigne all friction of the flow pipe
	friction();
	float sum = 0 ; 

	for(int i = 0 ; i < 25 ; i++){
		sum = 0 ;

		// create jocobian matrix and reisdual values in host
 		jaco(Q,h_r);
    	comR(h_r, Q);
    	// get the sum of residual value.
		for(int j = 0 ; j < N ; j++){
        	sum = sum + R[i]*R[i];
        }
        // Initialize all variable in pipe
		Init();
		// Send variables to the device
		Send_To_Device();
		// Do parallel computation in GPU
		Call_GPUSerial_Function();
		// Send variables from device to host
		Get_From_Device();
		// update flows 
		UpdateQ(Q,h_x);
	}
	t = clock() - t;

	// Print out all final value of each flow pipe
	for(int i = 0 ; i < N ; i++){
       	printf("Q[%d] = %g\n",i,Q[i]);
	}
	
	// print out total time for runing 
	printf("GPU takes %f second\n",((float)t/CLOCKS_PER_SEC));
	
	// Free memory in both device and host
	Free_Memory();
	return 0;
}

