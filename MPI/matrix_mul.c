#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


// TO RUN THIS CODE 

// Compile: mpicc file_name.c -o out
// Run: mpirun -n <No. of Process> ./out <No. of Matrices> <Rows In A> <Cols in A or Rows in B> <Cols in B>
// Example RUN: mpirun -n 5 100 3 3 3

void print_matrix(int rows, int cols, long long matrix[rows][cols]){
  //printf matrix
}

// Function to store the First Matrix of A,B and C = AxB into output file
void save_matrix(const char* filename, int rows, int cols, long long matrix[rows][cols]){
    FILE* file = fopen(filename, "w");
    if(file != NULL){
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                fprintf(file, "%3lld ", matrix[i][j]);
            }
            fprintf(file,"\n");
        }
    }
}

int main(int argc, char **argv){
    
    // MPI initialization----------Start
    MPI_Init(&argc, &argv);
    int size, rank;
    
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(argc != 5 ){
        if(rank == 0){
          printf("Worng command line args\nUsage:<num_matrices><rows_A><cols_A/rows_B><cols_B>\n");
          MPI_Finalize();
          return 1;
        }
    }
    // MPI initialization-----------End
    
    // Initializing matrix array and matrix dimensions 
    long long num_of_matrices = atoll(argv[1]);
    long long rows_A = atoll(argv[2]);
    long long cols_A = atoll(argv[3]);
    long long cols_B = atoll(argv[4]);
    
    // Boradcast number of matrices and matrix dimensions to all processes 
    MPI_Bcast(&num_of_matrices, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);  // sending the number of matrices to every process
    MPI_Bcast(&rows_A, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);           // sending the row count of matrix A to every process
    MPI_Bcast(&cols_A, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);           // Sending the column count of matrix A / row count of matrix B to all processes 
    MPI_Bcast(&cols_B, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);           // Sending the coulmn count of matrix B to all processes
    
    
    // Checking if the matrices can be divided equally among the processes
    if(num_of_matrices % size != 0){
        if(rank == 0){
            printf("Matrices can not be evenly divided among the processes\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    
    // Initializing matrices with random values ------------ Start
    long long A[num_of_matrices][rows_A][cols_A];
    long long B[num_of_matrices][cols_A][cols_B];
    long long C[num_of_matrices][rows_A][cols_B];
    
    if(rank == 0){
        for(int num = 0; num < num_of_matrices; num++){
            
            for(int i = 0; i < rows_A; i++){
              for(int j = 0; j < cols_A; j++){
                  long long x = rand() % 100;
                  A[num][i][j] = x;
              }
            }
            for(int i = 0; i < cols_A; i++){
              for(int j = 0; j < cols_B; j++){
                  long long x = rand() % 100;
                  B[num][i][j] = x;
              }
            }
        }
    }
    // Initializing matrices with random values ------------ End
    
    // Each process will have its own chunk of matrix. Creating matrix array so the chunks can be stored
    long long local_count = num_of_matrices/size;
    long long local_A[local_count][rows_A][cols_A];
    long long local_B[local_count][cols_A][cols_B];
    long long local_C[local_count][rows_A][cols_B];
    
    MPI_Scatter(A, local_count*rows_A*cols_A, MPI_LONG_LONG, local_A, 
                local_count*rows_A*cols_A, MPI_LONG_LONG, 0, MPI_COMM_WORLD
                ); // The array of A matrix has been divided into equal chunks and sent to all the processes
    MPI_Scatter(B, local_count*cols_A*cols_B, MPI_LONG_LONG, local_B, 
                local_count*cols_A*cols_B, MPI_LONG_LONG, 0, MPI_COMM_WORLD
                ); // // The array of A matrix has been divided into equal chunks and sent to all the processes
    
    MPI_Barrier(MPI_COMM_WORLD); // synchornizing so that every process has their chunks and are ready to perform multiplication
    
    
    
    // Time calculation and Matrix multiplication -------- Start
    double start_time = MPI_Wtime();
    
    // Matrix multiplication logic (Must implement on your own !!!!!!!!!!!!!)
    for(int num = 0; num < local_count; num++){
        for(int i = 0; i < rows_A; i++){
          for(int j = 0; j < cols_B; j++){
            local_C[num][i][j] = 0;
            for(int k = 0; k < cols_A; k++){
                local_C[num][i][j] += (local_A[num][i][k] * local_B[num][k][j]) % 100;
            }
            local_C[num][i][j] %= 100;
          }
        }
    }
    
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    
    double all_times[size];
    MPI_Gather(&local_time, 1, MPI_DOUBLE, all_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Gathering times from each process and storing it to an array
    
    double max_time;
    
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Calculating the maximum time 
    
    MPI_Gather(local_C, local_count*rows_A*cols_B, MPI_LONG_LONG, C, local_count*rows_A*cols_B, MPI_LONG_LONG, 0, MPI_COMM_WORLD); // Gathering results from all processes
    //  Time calculation and Matrix multiplication --------  End
    
    
    // Showing results
    if(rank == 0){
      printf("\n\tExecution Time per process\n");
      for(int i = 0; i < size; i++){
        printf("Process: %d\t time = %f seconds\n", i, all_times[i]);
      }
      
      printf("Maximum Time taken %f seconds\n", max_time);
      
      // Save first matrix result for verification
        save_matrix("A_0.txt", rows_A, cols_A, A[0]);
        save_matrix("B_0.txt", cols_A, cols_B, B[0]);
        save_matrix("C_0.txt", rows_A, cols_B, C[0]);
      
    }
    
    MPI_Finalize();
    return 0;
        
}




















