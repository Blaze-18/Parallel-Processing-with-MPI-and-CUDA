%%writefile matrix.cu
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// TO RUN THIS FILE (Run the commands in colab Cell)
// COMPILE: !nvcc -arch=sm_75 matrix.cu -o matrix
// RUN: !time ./matrix <No. of Process> <No of Matrices> <Rows in A> <Cols in A or Rows in B> <Cols in B>
// Example: !time ./matrix 128 200 3 3 3 (Shows output in colab cell)
// Example: !time ./matrix 128 200 3 3 3 > output128.txt  (Shows output in a text file named output128.txt)

// This function will run in GPU  (Function for performing matrix multiplication on the GPU)
__global__ void matrixMul(float *A, float *B, float *R, int M, int N, int P, int batchOffset) {
    int k = threadIdx.x + batchOffset; // Set the thread index using batch offset
    
    // Set pointers to the Matrix element
    float *a = A + k * M * N;
    float *b = B + k * N * P;
    float *r = R + k * M * P;
    // Matrix multiplication logic
  for(int outer = 0; outer < 100; outer++) {
    for(int i = 0; i < M; i++) {
      for(int l = 0; l < P; l++) {
        r[i * P + l] = 0.0f; // explicitly set to 0
        for(int j = 0; j < N; j++) {
          r[i * P + l] += a[i * N + j] * b[j * P + l];
        }
      }
    }
  }
  }

// print the first matrix only in the cell or to an output file
void printMatrix(float *A, int M, int N) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
          printf("%.0f ", A[i * N + j]);
        }
        cout<<endl;
    }
}



int main(int argc, char* argv[]) {

    // Taking the inputs from command line and making them integers
    int threads = atoi(argv[1]); // Number of threads
    int K = atoi(argv[2]); // Number of Matrices
    int M = atoi(argv[3]); // Rows of A
    int N = atoi(argv[4]); // Cols of A or Rows of B
    int P = atoi(argv[5]); // Cols of B

   
    // Calculating size of each matrix including the result
    int size_of_a = K * M * N;
    int size_of_b = K * N * P;
    int size_of_r = K * M * P;
    
    // Allocating memory in CPU for each matrix (Format is GPUT understandable)
    float *h_A = (float*)malloc(size_of_a * sizeof(float));
    float *h_B = (float*)malloc(size_of_b * sizeof(float));
    float *h_R = (float*)malloc(size_of_r * sizeof(float));
    
    // Intializing the matrices in a flattend format [A000, A001, A002 ......, B000, B001, B002....]
    for(int i = 0; i < size_of_a; i++) {
        h_A[i] = rand() % 10;
    }
    for(int i = 0; i < size_of_b; i++) {
        h_B[i] = rand() % 10;
    }
    
    
    // Allocating memory for GPU to copy the Initialized array to GPU (Sending matrix to the GPU)
    float *d_A;
    cudaMalloc(&d_A, size_of_a * sizeof(float)); // Allocating memory for A matrix in GPU
    cudaMemcpy(d_A, h_A, size_of_a * sizeof(float), cudaMemcpyHostToDevice); // Copying A matrix from CPU to GPU

    float *d_B;
    cudaMalloc(&d_B, size_of_b * sizeof(float)); 
    cudaMemcpy(d_B, h_B, size_of_b * sizeof(float), cudaMemcpyHostToDevice); // Copying A matrix from CPU to GPU

    float *d_R;
    cudaMalloc(&d_R, size_of_r * sizeof(float)); // Allocating memory for B matrix in GPU
    cudaMemset(d_R, 0, size_of_r * sizeof(float)); // Set the value 0 in C matrix [C000 = 0, C001 = 0, C002 = 0.....]


    // CODE FOR GPU execution is here(MUST BE DONE ON YOUR OWN)
    int remainingMatrices = K;
    int batchOffset = 0;
    // Configure threads and Batch offsets and Global function so that GPU can perform the multiplication
    while(remainingMatrices > 0) {
    // Inside the loop Threads perform matrix multiplication in batches (Batch Size = No of threads defined in the cmd line)
        int currentBatchSize = min(remainingMatrices, threads);
        matrixMul<<<1, currentBatchSize>>>(d_A, d_B, d_R, M, N, P, batchOffset);
        cudaDeviceSynchronize();
        remainingMatrices -= currentBatchSize;
        batchOffset += currentBatchSize;
    }
    
    // Send the Results back to the CPU for showing it to the user
    cudaMemcpy(h_R, d_R, size_of_r * sizeof(float), cudaMemcpyDeviceToHost);
    
    
    // SHowing output
    cout<<"Matrix A[0]:"<<endl;
    printMatrix(h_A, M, N);
    cout<<"Matrix B[0]:"<<endl;
    printMatrix(h_B, N, P);
    cout<<"Matrix R[0]:"<<endl;
    printMatrix(h_R, M, P);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_R);
    free(h_A);
    free(h_B);
    free(h_R);
    return 0;
}

