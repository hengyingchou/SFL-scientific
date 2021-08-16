#include"gpumain.h"
//Declare important variables
float *h_A;// Host variable
float *h_A2;
float *h_x;
float *h_x2;
float *h_B;
float *h_R;
float *h_scalar;
float A[N][N] = {0};
float *Q ;
float *h_r ;
float *R;
// Device variable
float *d_A;
float *d_A2;
float *d_B;
float *d_x;
float *d_x2;
float *d_R;
float *d_R2;
float *d_p;
float *d_p2;
float *d_Ap;
float *d_Ap2;
float *d_temp;
float *d_scalar;


__global__ void SetupCG(float *B, float *R, float *p,float *Ap, float *R2, float *p2, float *Ap2, int n);
__global__ void MatrixVectorProduct(float*A,float*P,float*Ap,int n);
__global__ void Dotproduct(float *A,float *P, float*temp , int n);
__global__ void Sum_and_store(float*temp,float*scalar,int n , int Blockspergrid);
__global__ void UpdateX(float *x, float *p, float *scalar ,int n);
__global__ void UpdateR(float *R, float *Ap, float *scalar ,int n);
__global__ void UpdateP(float *P, float *R, float *scalar ,int n);
__global__ void Updatebeta(float *scalar);
void Matrixtranspose(float*A , float*A2,int n);
void Allocate_Memory()
{

	// our N*N  variable A;
	size_t size = N*N*sizeof(float);
	h_A = (float*)malloc(size);
	h_A2 = (float*)malloc(size);
	cudaError_t  Error = cudaMalloc((void**)&d_A,size);
	printf("CUDA error(malloc d_a)=%s\n",cudaGetErrorString(Error)); 
	Error = cudaMalloc((void**)&d_A2,size);
        printf("CUDA error(malloc d_a2)=%s\n",cudaGetErrorString(Error));

	//our  1D (N) variable
	size = N*sizeof(float); 
	h_B = (float*)malloc(size);
	h_x = (float*)malloc(size);
	h_x2 = (float*)malloc(size);
	h_R = (float*)malloc(size);
	h_r =(float*)malloc(size);
	Q = (float*)malloc(size);
	R = (float*)malloc(size);
	for(int i = 0 ; i < N ; i++) Q[i] = 1.0;
	size = 5*sizeof(float);
	h_scalar = (float*)malloc(size);

	//Allocate device memory
	 size = N*sizeof(float);
	 Error = cudaMalloc((void**)&d_B,size); printf("CUDA error(malloc d_B)=%s\n",cudaGetErrorString(Error));
	 Error = cudaMalloc((void**)&d_x,size); printf("CUDA error(malloc d_x)=%s\n",cudaGetErrorString(Error));
	 Error = cudaMalloc((void**)&d_x2,size); printf("CUDA error(malloc d_x2)=%s\n",cudaGetErrorString(Error));
	 Error = cudaMalloc((void**)&d_R,size); printf("CUDA error(malloc d_R)=%s\n",cudaGetErrorString(Error));
	 Error = cudaMalloc((void**)&d_R2,size); printf("CUDA error(malloc d_R2)=%s\n",cudaGetErrorString(Error));
	 Error = cudaMalloc((void**)&d_p,size); printf("CUDA error(malloc d_p)=%s\n",cudaGetErrorString(Error));
	 Error = cudaMalloc((void**)&d_p2,size); printf("CUDA error(malloc d_p2)=%s\n",cudaGetErrorString(Error));
 	 Error = cudaMalloc((void**)&d_Ap,size); printf("CUDA error(malloc d_Ap)=%s\n",cudaGetErrorString(Error));
	 Error = cudaMalloc((void**)&d_Ap2,size); printf("CUDA error(malloc d_Ap2)=%s\n",cudaGetErrorString(Error));
	 Error = cudaMalloc((void**)&d_temp,size); printf("CUDA error(malloc d_temp)=%s\n",cudaGetErrorString(Error));
	// Small array holding our scalaers(RTR, RTR_new, PTAP, alpha, beta);
	 size = 5*sizeof(float);
	 Error = cudaMalloc((void**)&d_scalar,size); printf("CUDA error(malloc d_scalar)=%s\n",cudaGetErrorString(Error));
}

// sending Variable from host to device 
void Send_To_Device()
{
	size_t size = N*N*sizeof(float);
	cudaError_t Error;
	Error = cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);printf("CUDA error(memcpy h_A->d_A = %s )\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_A2,h_A2,size,cudaMemcpyHostToDevice);printf("CUDA error(memcpy h_A->d_A =%s )\n",cudaGetErrorString(Error));
	size = N*sizeof(float);
	Error = cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);printf("CUDA error(memcpy h_x->d_x = %s )\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_x2,h_x2,size,cudaMemcpyHostToDevice);printf("CUDA error(memcpy h_x->d_x = %s)\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);printf("CUDA error(memcpy h_B->d_a = %s )\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_R,h_R,size,cudaMemcpyHostToDevice);printf("CUDA error(memcpy h_R->d_R = %s )\n",cudaGetErrorString(Error));
	size = 5*sizeof(float);
	Error = cudaMemcpy(d_scalar,h_scalar,size,cudaMemcpyHostToDevice);printf("CUDA error(memcpy h_sclar->d_scalar = %s )\n",cudaGetErrorString(Error));

}

// sending Variable from devcie to Host
void Get_From_Device()
{
	size_t size = N*N*sizeof(float);
	cudaError_t Error;
	Error =cudaMemcpy(h_A,d_A,size,cudaMemcpyDeviceToHost);printf("CUDA error(mamcpy d_A->h_A)=%s\n",cudaGetErrorString(Error));
	Error =cudaMemcpy(h_A2,d_A2,size,cudaMemcpyDeviceToHost);printf("CUDA error(mamcpy d_A2->h_A2)=%s\n",cudaGetErrorString(Error));
	size = N*sizeof(float);
	Error =cudaMemcpy(h_x2,d_x2,size,cudaMemcpyDeviceToHost);printf("CUDA error(mamcpy d_A2->h_A = %s\n",cudaGetErrorString(Error));
	Error =cudaMemcpy(h_x,d_x,size,cudaMemcpyDeviceToHost);printf("CUDA error(mamcpy d_x->h_x)=%s\n",cudaGetErrorString(Error));
	Error =cudaMemcpy(h_B,d_B,size,cudaMemcpyDeviceToHost);printf("CUDA error(mamcpy d_B->h_B)=%s\n",cudaGetErrorString(Error));
	Error =cudaMemcpy(h_R,d_R,size,cudaMemcpyDeviceToHost);printf("CUDA error(mamcpy d_R->h_R)=%s\n",cudaGetErrorString(Error));
	size = 5*sizeof(float);
	Error =cudaMemcpy(h_scalar,d_scalar,size,cudaMemcpyDeviceToHost);printf("CUDA error(mamcpy d_scalar->h_scalar)=%s\n",cudaGetErrorString(Error));
}

// friction of all pipe
void friction(){
        h_r[0] = 1.0; h_r[1] = 1.0; h_r[2] = 5.0; h_r[3] = 5.0; h_r[4] =  1.0;
        h_r[5] = 3.0; h_r[6] = 1.0; h_r[7] = 1.0; h_r[8] = 3.0; h_r[9] =  1.0;
        h_r[10] = 1.0; h_r[11] = 2.0; h_r[12] = 3.0; h_r[13]= 3.0; h_r[14] = 1.0;
        h_r[15] = 1.0; h_r[16] = 1.0; h_r[17] = 3.0; h_r[18] = 2.0; h_r[19] = 1.0;
        h_r[20] = 2.0; h_r[21] = 5.0; h_r[22] = 2.0; h_r[23] = 1.0; h_r[24] = 5.0;
        h_r[25] = 2.0; h_r[26] = 1.0; h_r[27] = 2.0; h_r[28] = 1.0;

}
void jaco(float *x , float *r){
        float t = 2.0;
		//1-st
        A[0][1] = 2.0*r[1]*x[1]; A[0][4] = t*r[4]*x[4]; A[0][6] = t*r[6]*x[6];
        A[0][5] = -t*r[5]*x[5]; A[0][3] = -t*r[3]*x[3];
        //2-sec
        A[1][2] = t*r[2]*x[2]; A[1][8] = -t*r[8]*x[8]; A[1][7] = -t*r[7]*x[7];
        A[1][4] = -t*r[4]*x[4];
        //3-rd
        A[2][7] = t*r[7]*x[7]; A[2][10] = t*r[10]*x[10]; A[2][9] = -t*r[9]*x[9];
        A[2][6] = -t*r[6]*x[6];
        //4-th
        A[3][5] = t*r[5]*x[5]; A[3][9] = t*r[9]*x[9]; A[3][11] = t*r[11]*x[11];
        A[3][16] = t*r[16]*x[16]; A[3][17] = t*r[17]*x[17]; A[3][20] = -t*r[20]*x[20];
        A[3][15] = -t*r[15]*x[15];
        //5th
        A[4][8] = t*r[8]*x[8]; A[4][12] = t*r[12]*x[12]; A[4][13] = -t*r[13]*x[13];
        A[4][11] = -t*r[11]*x[11]; A[4][10] = -t*r[10]*x[10];
        //6-th
        A[5][13] = t*r[13]*x[13]; A[5][19] = -t*r[19]*x[19]; A[5][21] = -t*r[21]*x[21];
        A[5][18] = -t*r[18]*x[18]; A[5][16] = -t*r[16]*x[16];
        //7-th
        A[6][18] = t*r[18]*x[18]; A[6][23] = t*r[23]*x[23]; A[6][22] = -t*r[22]*x[22];
        A[6][17] = -t*r[17]*x[17];
		//8-th
        A[7][20] = t*r[20]*x[20]; A[7][22] = t*r[22]*x[22]; A[7][25] = t*r[25]*x[25];
        A[7][24] = -t*r[24]*x[24];
        //9-th
        A[8][21] = t*r[21]*x[21]; A[8][26] = t*r[26]*x[26]; A[8][27] = -t*r[27]*x[27];
        A[8][25] = -t*r[25]*x[25]; A[8][23] = -t*r[23]*x[23];
        //10-th
        A[9][24] = 1.0; A[9][25] = 1.0; A[9][27] = -1.0;
        //11-th
        A[10][0] = -t*r[0]*x[0]; A[10][1] = -t*r[1]*x[1]; A[10][2] = -t*r[2]*x[2];
        A[10][12] = -t*r[12]*x[12]; A[10][14] = -t*r[14]*x[14];
        //12-th
        A[11][0] = t*r[0]*x[0]; A[11][3] = t*r[3]*x[3]; A[11][15] = t*r[15]*x[15];
        A[11][24] = t*r[24]*x[24]; A[11][27] = t*r[27]*x[27]; A[11][28] = t*r[28]*x[28];
        //13-th
        A[12][0] = 1.0; A[12][1] = -1.0; A[12][3] = -1.0;
        //14-th
        A[13][1] = 1.0; A[13][2] = -1.0; A[13][4] = -1.0;
        //15-th
        A[14][4] = 1.0; A[14][6] = -1.0; A[14][7] = -1.0;
        //16-th
        A[15][3] = 1.0; A[15][5] = -1.0; A[15][15] = -1.0;
        //17-th
        A[16][5] = 1.0; A[16][6] = 1.0; A[16][9] = -1.0;
        //18-th
        A[17][7] = 1.0; A[17][8] = -1.0; A[17][10] = -1.0;
        //19-th
        A[18][2] = 1.0; A[18][8] = 1.0; A[18][12] = -1.0;
        //20-th
        A[19][9] = 1.0; A[19][10] = 1.0; A[19][11] = -1.0;
        //21-th
        A[20][11] = 1.0; A[20][13] = -1.0; A[20][16] = -1.0;
        //22-th
        A[21][12] = 1.0; A[21][13] = 1.0; A[21][19] = 1.0; A[21][14] = -1.0;
        //23-th
        A[22][16] = 1.0; A[22][17] = -1.0; A[22][18] = -1.0;
 		//24-th
        A[23][15] = 1.0; A[23][20] = -1.0; A[23][24] = -1.0;
        //25-th
        A[24][20] = 1.0; A[24][17] = 1.0; A[24][22] = -1.0;
        //26-th
        A[25][18] = 1.0; A[25][21] = -1.0; A[25][23] = -1.0;
        //27-th
        A[26][21] = 1.0; A[26][19] = -1.0; A[26][26] = -1.0;
        //28-th
        A[27][26] = 1.0; A[27][27] = 1.0; A[27][28] = -1.0;
        //29-th
        A[28][22] = 1.0; A[28][23] = 1.0; A[28][25] = -1.0;
}

// compute residual matrix 
void comR(float *d_r, float *x)
{
        R[0] = d_r[1]*x[1]*fabs(x[1])+d_r[4]*x[4]*fabs(x[4])+d_r[6]*x[6]*fabs(x[6])-d_r[5]*x[5]*fabs(x[5])-d_r[3]*x[3]*fabs(x[3]);
        R[1] = d_r[2]*x[2]*fabs(x[2])-d_r[8]*x[8]*fabs(x[8])-d_r[7]*x[7]*fabs(x[7])-d_r[4]*x[4]*fabs(x[4]);
        R[2] = d_r[7]*x[7]*fabs(x[7])+d_r[10]*x[10]*fabs(x[10])-d_r[9]*x[9]*fabs(x[9])-d_r[6]*x[6]*fabs(x[6]);
        R[3] = d_r[5]*x[5]*fabs(x[5])+d_r[9]*x[9]*fabs(x[9])+d_r[11]*x[11]*fabs(x[11])+d_r[16]*x[16]*fabs(x[16])+d_r[17]*x[17]*fabs(x[17])-d_r[20]*x[20]*fabs(x[20])-d_r[15]*x[15]*fabs(x[15]);
        R[4] = d_r[8]*x[8]*fabs(x[8])+d_r[12]*x[12]*fabs(x[12])-d_r[13]*x[13]*fabs(x[13])-d_r[11]*x[11]*fabs(x[11])-d_r[10]*x[10]*fabs(x[10]);
        R[5] = d_r[13]*x[13]*fabs(x[13])-d_r[19]*x[19]*fabs(x[19])-d_r[21]*x[21]*fabs(x[21])-d_r[18]*x[18]*fabs(x[18])-d_r[16]*x[16]*fabs(x[16]);
        R[6] = d_r[18]*x[18]*fabs(x[18])+d_r[23]*x[23]*fabs(x[23])-d_r[22]*x[22]*fabs(x[22])-d_r[17]*x[17]*fabs(x[17]);
        R[7] = d_r[22]*x[22]*fabs(x[22])+d_r[25]*x[25]*fabs(x[25])-d_r[24]*x[24]*fabs(x[24])+d_r[20]*x[20]*fabs(x[20]);
        R[8] = d_r[21]*x[21]*fabs(x[21])+d_r[26]*x[26]*fabs(x[26])-d_r[27]*x[27]*fabs(x[27])-d_r[25]*x[25]*fabs(x[25])-d_r[23]*x[23]*fabs(x[23]);
        R[9] = x[25]+x[24]-x[27];
        R[10] = 100.0-d_r[0]*x[0]*fabs(x[0])-d_r[1]*x[1]*fabs(x[1])-d_r[2]*x[2]*fabs(x[2])-d_r[12]*x[12]*fabs(x[12])-d_r[14]*x[14]*fabs(x[14]);
        R[11] = d_r[0]*x[0]*fabs(x[0])+d_r[3]*x[3]*fabs(x[3])+d_r[15]*x[15]*fabs(x[15])+d_r[24]*x[24]*fabs(x[24])+d_r[27]*x[27]*abs(x[27])+d_r[28]*x[28]*fabs(x[28])-100.0;
        R[12] = x[0]-x[1]-x[3];
        R[13] = x[1]-x[4]-x[2];
        R[14] = x[4]-x[6]-x[7];
        R[15] = x[3]-x[5]-x[15];
        R[16] = x[5]+x[6]-x[9];
        R[17] = x[7]-x[8]-x[10];
        R[18] = x[2]+x[8]-x[12];
        R[19] = x[9]+x[10]-x[11];
        R[20] = x[11]-x[13]-x[16];
        R[21] = x[12]+x[13]+x[19]-x[14];
        R[22] = x[16]-x[17]-x[18];
        R[23] = x[15]-x[20]-x[24];
        R[24] = x[20]+x[17]-x[22];
        R[25] = x[18]-x[21]-x[23];
        R[26] = x[21]-x[19]-x[26];
        R[27] = x[26]+x[27]-x[28];
        R[28] = x[22]+x[23]-x[25];
}
void UpdateQ(float *x ,float*dx)
{
        for(int i = 0 ; i < N ; i++){
        x[i] = x[i] - 0.5* dx[i];
        }
}

void Init(){

	for(int i = 0 ; i < N ;i ++){
		for(int j = 0 ; j < N ; j ++){
		h_A[i*N+j] = A[i][j];
		}
		h_B[i] = R[i];
        //X
        h_x[i] = 0.0;
	h_x2[i] = 0.0;
        //R
        h_R[i] = 0.0;
	}
        //scalar
	for(int i = 0; i < 5 ; i++) h_scalar[i] = 1.0;
	// make transponse
	Matrixtranspose(h_A , h_A2, N);
}
void Matrixtranspose(float*A , float*A2,int n)
{
        for(int i = 0 ; i < n ; i++){
        for(int j = 0 ; j < n ; j++){
        A2[i*N+j] = A[j*N+i];
        }
	}
}

void Call_GPUSerial_Function()
{
	int threadsperblock = 8;
    int blockspergrid = (N+ threadsperblock -1 )/threadsperblock;
	//Prepare for setup
	MatrixVectorProduct<<<blockspergrid,threadsperblock>>>(d_A,d_x,d_Ap,N);
	MatrixVectorProduct<<<blockspergrid,threadsperblock>>>(d_A2,d_x2,d_Ap2,N);
	// Set up CG 
	SetupCG<<<blockspergrid,threadsperblock>>>(d_B,d_R,d_p,d_Ap,d_R2,d_p2,d_Ap2,N);
	for(int i = 0 ; i < 3*N ; i++ ){
	//Computr AP
	MatrixVectorProduct<<<blockspergrid,threadsperblock>>>(d_A,d_p,d_Ap,N);
	MatrixVectorProduct<<<blockspergrid,threadsperblock>>>(d_A2,d_p2,d_Ap2,N);
	//Compute PTAP
	Dotproduct<<<blockspergrid,threadsperblock>>>(d_Ap,d_p2,d_temp,N);
	Sum_and_store<<<1,1>>>(d_temp,d_scalar,2, blockspergrid);
	//Compute RTR
	Dotproduct<<<blockspergrid,threadsperblock>>>(d_R,d_R2,d_temp,N);
    Sum_and_store<<<1,1>>>(d_temp,d_scalar,0,blockspergrid);
	//Update X
	UpdateX<<<blockspergrid,threadsperblock>>>(d_x,d_p,d_scalar,N);
	UpdateX<<<blockspergrid,threadsperblock>>>(d_x2,d_p2,d_scalar,N);
	//Update R
	UpdateR<<<blockspergrid,threadsperblock>>>(d_R, d_Ap,d_scalar,N);
	UpdateR<<<blockspergrid,threadsperblock>>>(d_R2, d_Ap2,d_scalar,N);
	//compute new resiudal
	Dotproduct<<<blockspergrid,threadsperblock>>>(d_R,d_R2,d_temp,N);
	Sum_and_store<<<1,1>>>(d_temp,d_scalar,1, blockspergrid);
	//Update P ( and compute beta)
	UpdateP<<<blockspergrid,threadsperblock>>>(d_p, d_R, d_scalar,N);
	UpdateP<<<blockspergrid,threadsperblock>>>(d_p2, d_R2, d_scalar,N);
	//pass the residual to  host 
	//cudaError_t Error = cudaMemcpy(h_scalar,d_scalar, 5*sizeof(float) , cudaMemcpyDeviceToHost); 
	//printf("Interation %i = RTR = %g, RTR_new = %g,PTAP = %g, alpha = %g, beta = %g\n",i,h_scalar[0],h_scalar[1],h_scalar[2],h_scalar[3],h_scalar[4]);
	Updatebeta<<<blockspergrid,threadsperblock>>>(d_scalar);
}
}
__global__ void SetupCG(float *B, float *R, float *p,float *Ap, float *R2, float *p2, float *Ap2,int n){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i < n)
	{
	R[i] = B[i];
	p[i] = R[i];

	R2[i] = B[i];
        p2[i] = R2[i];
	}
}

// Matrix and vector dot inner prodoct
__global__ void MatrixVectorProduct(float*A,float*P,float*Ap,int n)
{
	__shared__ float sum[N];
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i < n)
	{
		sum[i] = 0;
		for(int k = 0 ; k < n ; k++ ){
		sum[i] = sum[i] + A[i*n+k]*P[k] ;
		}
		Ap[i] = sum[i];
	}
}

// Vector Dot product
__global__ void Dotproduct(float *A,float *B, float*z , int n)
{
	__shared__ float temp[BPG];
        int I = threadIdx.x;
        int i = I + blockDim.x*blockIdx.x;

        if(i < n) {
	temp[I] = A[i]*B[i];
	z[i] = 0;
	}else{
	temp[I] = 0;
	}
        __syncthreads();

        for(int stride = blockDim.x/2 ; stride>0 ; stride = stride/2)
        {
        if(I < stride){
                temp[I] += temp[I+stride];
	}
        __syncthreads();
	}
	if(I == 0 ) z[blockIdx.x] = temp[0]; 
}


__global__ void Sum_and_store(float*temp,float*scalar,int n ,int  Blockspergrid)
{

	float sum = 0;
	for(int i = 0 ; i < Blockspergrid ; i++)
	sum = temp[i] + sum;
	scalar[n] = sum;
}
__global__ void UpdateX(float *x, float *p, float *scalar ,int n){

	int i = threadIdx.x + blockDim.x*blockIdx.x;
	scalar[3] = (scalar[0]/scalar[2]);
	if(i < n){
	x[i] = x[i] + p[i]*(scalar[3]);
	}
	__syncthreads();
}

__global__ void UpdateR(float *R, float *Ap, float *scalar ,int n){

        int i = threadIdx.x + blockDim.x*blockIdx.x;
        if(i < n){
        R[i] = R[i] - Ap[i]*(scalar[3]);
        }
        __syncthreads();
}

__global__ void UpdateP(float *P, float *R, float *scalar ,int n){

        int i = threadIdx.x + blockDim.x*blockIdx.x;
	scalar[4] = scalar[1]/scalar[0]; 
        if(i < n){
        P[i] = R[i] + P[i]*scalar[4];
        }
        __syncthreads();
}
__global__ void Updatebeta(float *scalar){
scalar[0] = scalar[1];
}

void Free_Memory(){
	// free host memory 
	 free(h_A);
	 free(h_A2);
	 free(h_x);
	 free(h_x2);
	 free(h_B);
     free(h_R);
     free(h_scalar);
	 free(h_r);
 	 free(Q);
	 free(R);
	 // free device memory
	 cudaFree(d_A);
	 cudaFree(d_A2);
	 cudaFree(d_B);
	 cudaFree(d_x);
	 cudaFree(d_x2);
	 cudaFree(d_R);
	 cudaFree(d_R2);
	 cudaFree(d_p);
	 cudaFree(d_p2);
	 cudaFree(d_Ap);
	 cudaFree(d_Ap2);
	 cudaFree(d_temp);
	 cudaFree(d_scalar);

}






