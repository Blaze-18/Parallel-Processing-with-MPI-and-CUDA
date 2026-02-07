
%%writefile search_phonebook.cu

/*
    Question: Consider a phonebook is given as a text file and a person name P. 
    Write a program using CUDA to search for a person P (given as a string) in the phonebook. 
    The program must return the matching name and matching phone number of the person P from the phonebook, 
    along with the total corresponding and correct numbers. 
    
    Input: No. of CPU cores, (phonebook from file), person name P 
    Output: Total searching time, matching names and correct numbers


    To Run this program:
    1. Copy the code into a colab cell
    2. Change runtime to t4 GPU
    3. Upload text file to the sample data folder
    4. In new Cell ---
        Compile: nvcc -arch=sm_75 cuda_file_name.cu -o search_phonebook
        Run Format: !time ./search_phonebook <phonebook.txt> <Person_Name> <threads_per_block>
        Example runs:
        With 128 threads
        !time ./search_phonebook phonebook.txt SHAHRIAR 128
        with 256 threads
        !time ./search_phonebook phonebook.txt SHAHRIAR 256
*/


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

    ifstream file(file_name);
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_name << endl;
        return false;
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;

        int pos = line.find(",");
        if (pos == string::npos) continue;

        string name = line.substr(1, pos - 2);
        string number = line.substr(pos + 2, line.size() - pos - 3);

        names.push_back(name);
        numbers.push_back(number);
    }

    file.close();
    return !names.empty();
}



// STEP 7 -------------------- Device substring match function
__device__ bool check(const char* str1, const char* str2, int len) {
    for (int i = 0; str1[i] != '\0'; i++) {
        int j = 0;
        while (str1[i + j] != '\0' && j < len && str1[i + j] == str2[j]) {
            j++;
        }
        if (j == len) return true;
    }
    return false;
}



// STEP 7 -------------------- CUDA kernel for parallel search
__global__ void searchPhonebook(
    char* d_names,
    int num_contacts,
    char* search_name,
    int search_len,
    int* d_results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_contacts) {
        char* current_name = d_names + idx * MAX_STR_LEN;
        d_results[idx] = check(current_name, search_name, search_len) ? 1 : 0;
    }
}



// ============================ MAIN ============================
int main(int argc, char* argv[]) {

    // STEP 1 -------------------- Read command-line arguments
    if (argc != 4) {
        cerr << "Usage: " << argv[0]
            << " <phonebook.txt> <search_string> <threads_per_block>\n";
        return 1;
    }

    // Base path where phonebook exists in Colab
    string base_path = "/content/sample_data/";

    // Final full path to phonebook file
    string file_name = base_path + string(argv[1]);

    cout << "file path: " << file_name << endl;

    string search_string = argv[2];
    int threads_per_block = atoi(argv[3]);

    // Replace '_' with space
    replace(search_string.begin(), search_string.end(), '_', ' ');


    // STEP 2 -------------------- Load phonebook
    vector<string> host_names_vec;
    vector<string> host_numbers_vec;

    if (!parsePhonebook(file_name, host_names_vec, host_numbers_vec)) {
        cerr << "No contacts found.\n";
        return 1;
    }

    int num_contacts = host_names_vec.size();



    // STEP 3 -------------------- Prepare host memory
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



    // STEP 7 -------------------- Start timing & launch kernel
    auto start = chrono::high_resolution_clock::now();

    searchPhonebook<<<blocks, threads_per_block>>>(
        d_names, num_contacts, d_search_name, search_len, d_results
    );

    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();
    double elapsed =
        chrono::duration<double, milli>(end - start).count();



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
    cout << "\nMatching Contacts:\n";
    for (const auto& c : matched_contacts) {
        cout << c.name << " " << c.number << endl;
    }

    cout << "\nTotal Matches: " << matched_contacts.size() << endl;
    cout << "Total Searching Time (ms): " << elapsed << endl;



    // STEP 12 -------------------- Cleanup
    free(h_names);
    free(h_results);
    cudaFree(d_names);
    cudaFree(d_results);
    cudaFree(d_search_name);

    return 0;
}
