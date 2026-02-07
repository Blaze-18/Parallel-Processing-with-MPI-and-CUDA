%%writefile search_phonebook.cu

// STEP 0 -------------------- Include libraries & definitions
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define MAX_STR_LEN 50   // Fixed GPU storage size for each name



// STEP 10 -------------------- Struct for CPU sorting
struct ResultContact {
    string name;
    string number;

    bool operator<(const ResultContact& other) const {
        return name < other.name;   // Ascending alphabetical order
    }
};



// STEP 2 -------------------- Parser function (CPU file reading)
bool parsePhonebook(const string& file_name,
                    vector<string>& names,
                    vector<string>& numbers) {

    ifstream file(file_name);                 // Open phonebook file
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_name << endl;
        return false;
    }

    string line;
    while (getline(file, line)) {             // Read file line-by-line
        if (line.empty()) continue;

        int pos = line.find(",");
        if (pos == string::npos) continue;

        string name = line.substr(1, pos - 2);                      // Extract name
        string number = line.substr(pos + 2, line.size() - pos - 3); // Extract number

        names.push_back(name);
        numbers.push_back(number);
    }

    file.close();
    return !names.empty();                    // Return success if contacts exist
}



// STEP 7 -------------------- Device substring match function
__device__ bool check(const char* str1, const char* str2, int len) {
    for (int i = 0; str1[i] != '\0'; i++) {
        int j = 0;
        while (str1[i + j] != '\0' && j < len && str1[i + j] == str2[j]) {
            j++;
        }
        if (j == len) return true;            // Full match found
    }
    return false;                             // No match
}



// STEP 7 -------------------- CUDA kernel for parallel search
__global__ void searchPhonebook(
    char* d_names,        // GPU names array
    int num_contacts,     // Total contacts
    char* search_name,    // GPU search string
    int search_len,       // Length of search string
    int* d_results        // Output match flags
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // Global thread index

    if (idx < num_contacts) {
        char* current_name = d_names + idx * MAX_STR_LEN;  // One thread → one contact
        d_results[idx] = check(current_name, search_name, search_len) ? 1 : 0;
    }
}



// ============================ MAIN ============================
int main(int argc, char* argv[]) {

    // STEP 1 -------------------- Read command-line arguments
    if (argc != 3) {
        cerr << "Usage: " << argv[0]
             << " <search_string> <threads_per_block>\n";
        return 1;
    }

    string search_string = argv[1];        // User search input
    int threads_per_block = atoi(argv[2]);

    // STEP 1 (Fix) -------------------- Replace '_' with space
    replace(search_string.begin(), search_string.end(), '_', ' ');



    // STEP 2 -------------------- Load phonebook using parser
    string file_name = "/content/sample_data/phonebook1.txt";

    vector<string> host_names_vec;
    vector<string> host_numbers_vec;
    
    if (!parsePhonebook(file_name, host_names_vec, host_numbers_vec)) {
        cerr << "No contacts found.\n";
        return 1;
    }

    int num_contacts = host_names_vec.size();



    // STEP 3 -------------------- Prepare host memory for GPU
    char* h_names = (char*)malloc(num_contacts * MAX_STR_LEN);
    int* h_results = (int*)malloc(num_contacts * sizeof(int));

    for (int i = 0; i < num_contacts; i++) {
        strncpy(h_names + i * MAX_STR_LEN,
                host_names_vec[i].c_str(),
                MAX_STR_LEN - 1);
        h_names[i * MAX_STR_LEN + MAX_STR_LEN - 1] = '\0';
    }



    // STEP 4 -------------------- Allocate GPU memory
    char *d_names, *d_search_name;
    int* d_results;

    int search_len = search_string.length();

    cudaMalloc(&d_names, num_contacts * MAX_STR_LEN);
    cudaMalloc(&d_results, num_contacts * sizeof(int));
    cudaMalloc(&d_search_name, search_len + 1);



    // STEP 5 -------------------- Copy CPU → GPU
    cudaMemcpy(d_names, h_names,
               num_contacts * MAX_STR_LEN,
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_search_name, search_string.c_str(),
               search_len + 1,
               cudaMemcpyHostToDevice);



    // STEP 6 -------------------- Configure CUDA grid
    int blocks = (num_contacts + threads_per_block - 1) / threads_per_block;



    // STEP 7 -------------------- Launch CUDA kernel
    searchPhonebook<<<blocks, threads_per_block>>>(
        d_names, num_contacts, d_search_name, search_len, d_results
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel error: "
             << cudaGetErrorString(err) << endl;
        return 1;
    }

    cudaDeviceSynchronize();   // Wait for GPU completion



    // STEP 8 -------------------- Copy results GPU → CPU
    cudaMemcpy(h_results, d_results,
               num_contacts * sizeof(int),
               cudaMemcpyDeviceToHost);



    // STEP 9 -------------------- Collect matched contacts
    vector<ResultContact> matched_contacts;
    for (int i = 0; i < num_contacts; i++) {
        if (h_results[i] == 1) {
            matched_contacts.push_back({
                host_names_vec[i],
                host_numbers_vec[i]
            });
        }
    }



    // STEP 10 -------------------- Sort alphabetically
    sort(matched_contacts.begin(), matched_contacts.end());



    // STEP 11 -------------------- Print results
    cout << "\nSearch Results (Ascending Order):\n";
    for (const auto& c : matched_contacts) {
        cout << c.name << " " << c.number << endl;
    }



    // STEP 12 -------------------- Cleanup memory
    free(h_names);
    free(h_results);
    cudaFree(d_names);
    cudaFree(d_results);
    cudaFree(d_search_name);

    return 0;   // Program finished
}
// TO run this file in colab
// First execute this cell
// Run the following commands in new cells
// Compile: !nvcc -arch=sm_75 search_phonebook.cu -o out
// RUN: !time ./out <Search_String> <NO. of Threads per block>
// Example Run: !time ./out ANTU 128
// Example Run: !time ./out ANTU_RANI 256
// Example RUN: !time ./out SHAHRIAR 128 > output.txt
// This saves the output/results in output.txt file
