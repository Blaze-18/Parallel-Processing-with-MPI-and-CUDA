%%writefile matrix.cu
#include <iostream>
#include <cuda_runtime.h>

using namespace std;


/*
    Question: 
    Write a program using CUDA to compute C[1:k-A[1:k] x B/1:k]. 
    Here A is an array of matrices with dimension m x n and B is an array of matrices with dimension n x p. 
    And k is the size of array.
    
    
    Input: No. of threads, k, m, n, p
    Output: Execution Time, A[0], B[0], C[0]

    // TO RUN THIS FILE (Run the commands in colab Cell)
// COMPILE: !nvcc -arch=sm_75 matrix.cu -o matrix
// RUN: !time ./matrix <No. of Process> <No of Matrices> <Rows in A> <Cols in A or Rows in B> <Cols in B>
// Example: !time ./matrix 128 200 3 3 3 (Shows output in colab cell)
// Example: !time ./matrix 128 200 3 3 3 > output128.txt  (Shows output in a text file named output128.txt)


*/



// Step 8-------– Perform matrix multiplication in GPU threads
// This function will run in GPU  (Function for performing matrix multiplication on the GPU)
__global__ void matrixMul(float *A, float *B, float *R, int M, int N, int P, int batchOffset) {
    int k = threadIdx.x + batchOffset; // Set the thread index using batch offset
    
    // Set pointers to the Matrix element
    float *a = A + k * M * N;
    float *b = B + k * N * P;
    float *r = R + k * M * P;
    
    // Matrix multiplication logic
    // Outer loop repeats 100 times to increase workload for timing
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

// Step 10--------– Print first matrices and result (helper function)
void printMatrix(float *A, int M, int N) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
          printf("%.0f ", A[i * N + j]);
        }
        cout<<endl;
    }
}

int main(int argc, char* argv[]) {
    // Step 1 ---------– Read command-line inputs
    // Taking the inputs from command line and making them integers
    int threads = atoi(argv[1]); // Number of threads
    int K = atoi(argv[2]); // Number of Matrices
    int M = atoi(argv[3]); // Rows of A
    int N = atoi(argv[4]); // Cols of A or Rows in B
    int P = atoi(argv[5]); // Cols of B

    // Step 2 ----------– Compute total memory sizes
    // Calculating size of each matrix including the result
    int size_of_a = K * M * N;
    int size_of_b = K * N * P;
    int size_of_r = K * M * P;
    
    // Step 3 ------------– Allocate memory on CPU
    // Allocating memory in CPU for each matrix (Format is GPU understandable)
    float *h_A = (float*)malloc(size_of_a * sizeof(float));
    float *h_B = (float*)malloc(size_of_b * sizeof(float));
    float *h_R = (float*)malloc(size_of_r * sizeof(float));
    
    // Step 4 ----------------– Initialize matrices with random values
    // Intializing the matrices in a flattened format [A000, A001, A002 ......, B000, B001, B002....]
    for(int i = 0; i < size_of_a; i++) {
        h_A[i] = rand() % 10;
    }
    for(int i = 0; i < size_of_b; i++) {
        h_B[i] = rand() % 10;
    }
    
    // Step 5 ---------------– Allocate memory on GPU
    // Allocating memory for GPU to copy the Initialized array to GPU (Sending matrix to the GPU)
    float *d_A;
    cudaMalloc(&d_A, size_of_a * sizeof(float)); // Allocating memory for A matrix in GPU
    
    // Step 6 ---------------– Copy matrices from CPU → GPU
    cudaMemcpy(d_A, h_A, size_of_a * sizeof(float), cudaMemcpyHostToDevice); // Copying A matrix from CPU to GPU

    float *d_B;
    cudaMalloc(&d_B, size_of_b * sizeof(float)); 
    cudaMemcpy(d_B, h_B, size_of_b * sizeof(float), cudaMemcpyHostToDevice); // Copying B matrix from CPU to GPU

    float *d_R;
    cudaMalloc(&d_R, size_of_r * sizeof(float)); // Allocating memory for result matrix in GPU
    cudaMemset(d_R, 0, size_of_r * sizeof(float)); // Initialize result matrix to zeros

    // Step 7 ----------------– Launch CUDA kernel in batches
    // CODE FOR GPU execution is here (MUST BE DONE ON YOUR OWN)
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
    
    // Step 9 -------------------– Copy result from GPU → CPU
    // Send the Results back to the CPU for showing it to the user
    cudaMemcpy(h_R, d_R, size_of_r * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Step 10 --------------------– Print first matrices and result
    // Showing output
    cout<<"Matrix A[0]:"<<endl;
    printMatrix(h_A, M, N);
    cout<<"Matrix B[0]:"<<endl;
    printMatrix(h_B, N, P);
    cout<<"Matrix R[0]:"<<endl;
    printMatrix(h_R, M, P);
    
    // Step 11 -----------------– Free GPU and CPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_R);
    free(h_A);
    free(h_B);
    free(h_R);
    
    return 0;
}
