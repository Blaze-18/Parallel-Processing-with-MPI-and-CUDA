#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

/*
    Question:
     Consider phonebooks given as text files and a phone number P. Write a program using MPI 
     to search for the persons' names who's contact phone number is P in the phonebooks. 
     The program will generate an output file containing the line number (within input file) 
     and persons' names with phone number P. 
     
     Input: No. of processes, phone number P
    Output: Execution Time, a text file containing the line number and names with phone number P in ascending order of name

*/

// TO RUN THIS CODE: 
// Compile: mpic++ phonebook_search.cpp -o search
// RUN: mpirun -n <No. of processes> ./search <phonebook_file> <phone_number>
// Example: mpirun -n 4 ./search phonebook1.txt "012 01 123"

// Function to send a large string over MPI
void send_string(const string &text, int receiver) {
    int len = text.size() + 1;
    MPI_Send(&len, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);
    MPI_Send(text.c_str(), len, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

// Function to receive a large string over MPI
string receive_string(int sender) {
    int len;
    MPI_Status status;

    MPI_Recv(&len, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, &status);
    
    vector<char> buf(len);
    MPI_Recv(buf.data(), len, MPI_CHAR, sender, 1, MPI_COMM_WORLD, &status);

    return string(buf.data());
}

// Converts a range of a vector of strings into one single string for transmission
string vector_to_string(const vector<pair<int, string>> &data) {
    string result;
    for (const auto &item : data) {
        result += to_string(item.first) + "|" + item.second + "\n";
    }
    return result;
}

// Splits a large received string back into a vector of (line_number, name) pairs
vector<pair<int, string>> string_to_vector(const string &text) {
    vector<pair<int, string>> result;
    istringstream iss(text);
    string line;

    while (getline(iss, line)) {
        if (!line.empty()) {
            size_t pos = line.find('|');
            if (pos != string::npos) {
                int line_num = stoi(line.substr(0, pos));
                string name = line.substr(pos + 1);
                result.emplace_back(line_num, name);
            }
        }
    }
    return result;
}

// Reads phonebook file and returns all lines
vector<string> read_phonebook(const string &filename) {
    vector<string> lines;
    ifstream f(filename);
    
    if (!f.is_open()) {
        cerr << "Could not open file: " << filename << endl;
        return lines;
    }

    string line;
    int line_count = 1;
    while (getline(f, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
        line_count++;
    }
    f.close();
    return lines;
}

// Parse a phonebook line and extract name and phone number
pair<string, string> parse_line(const string &line) {
    // Format: "FirstName MiddleName LastName","012 01 123"
    size_t quote1 = line.find('"');
    size_t quote2 = line.find('"', quote1 + 1);
    size_t quote3 = line.find('"', quote2 + 1);
    size_t quote4 = line.find('"', quote3 + 1);
    
    if (quote4 == string::npos) {
        return {"", ""}; // Invalid format
    }
    
    string name = line.substr(quote1 + 1, quote2 - quote1 - 1);
    string phone = line.substr(quote3 + 1, quote4 - quote3 - 1);
    
    return {name, phone};
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            cerr << "Usage: mpirun -n <procs> " << argv[0]
                 << " <phonebook_file> <phone_number>\n";
            cerr << "Example: mpirun -n 4 ./search phonebook.txt \"012 01 123\"\n";
        }
        MPI_Finalize();
        return 1;
    }

    string filename = argv[1];
    string target_phone = argv[2];
    double start_time, end_time;

    if (rank == 0) {
        // ===== MASTER PROCESS =====
        
        // Read all lines from phonebook
        vector<string> all_lines = read_phonebook(filename);
        int total_lines = all_lines.size();
        
        if (total_lines == 0) {
            cout << "Phonebook is empty or file not found!" << endl;
            MPI_Finalize();
            return 0;
        }

        // Calculate chunk size for each process
        int chunk_size = (total_lines + size - 1) / size;
        
        // Send chunks to worker processes
        for (int i = 1; i < size; i++) {
            int start_idx = i * chunk_size;
            int end_idx = min((i + 1) * chunk_size, total_lines);
            
            // Send the range of lines as a single string
            string chunk_data;
            for (int j = start_idx; j < end_idx; j++) {
                chunk_data += to_string(j + 1) + ":" + all_lines[j] + "\n";
            }
            send_string(chunk_data, i);
        }

        start_time = MPI_Wtime();

        // Master searches its own chunk
        vector<pair<int, string>> master_results;
        int master_end = min(chunk_size, total_lines);
        
        for (int i = 0; i < master_end; i++) {
            auto [name, phone] = parse_line(all_lines[i]);
            if (!phone.empty() && phone == target_phone) {
                master_results.emplace_back(i + 1, name);
            }
        }

        // Receive results from workers
        vector<pair<int, string>> all_results = master_results;
        
        for (int i = 1; i < size; i++) {
            string received = receive_string(i);
            vector<pair<int, string>> worker_results = string_to_vector(received);
            all_results.insert(all_results.end(), 
                               worker_results.begin(), 
                               worker_results.end());
        }

        // Sort results by name (ascending order)
        sort(all_results.begin(), all_results.end(), 
             [](const pair<int, string> &a, const pair<int, string> &b) {
                 return a.second < b.second;
             });

        end_time = MPI_Wtime();

        // Write results to output file
        ofstream out("phonebook_results.txt");
        out << "Phone Number Search Results for: " << target_phone << "\n";
        out << "===========================================\n";
        
        if (all_results.empty()) {
            out << "No matches found.\n";
        } else {
            for (const auto &result : all_results) {
                out << "Line " << result.first << ": " << result.second << "\n";
            }
        }
        
        out << "\nTotal matches found: " << all_results.size() << "\n";
        out.close();

        // Display execution time
        cout << "\n=== Search Results ===\n";
        cout << "Phone Number: " << target_phone << endl;
        cout << "Total matches found: " << all_results.size() << endl;
        printf("Execution Time: %.6f seconds\n", end_time - start_time);
        cout << "Results saved to: phonebook_results.txt\n";

    } else {
        // ===== WORKER PROCESSES =====
        
        // Receive chunk from master
        string chunk_data = receive_string(0);
        
        start_time = MPI_Wtime();

        // Parse chunk data and search
        vector<pair<int, string>> worker_results;
        istringstream iss(chunk_data);
        string line_with_num;
        
        while (getline(iss, line_with_num)) {
            if (!line_with_num.empty()) {
                size_t colon_pos = line_with_num.find(':');
                if (colon_pos != string::npos) {
                    int line_num = stoi(line_with_num.substr(0, colon_pos));
                    string line = line_with_num.substr(colon_pos + 1);
                    
                    auto [name, phone] = parse_line(line);
                    if (!phone.empty() && phone == target_phone) {
                        worker_results.emplace_back(line_num, name);
                    }
                }
            }
        }

        end_time = MPI_Wtime();

        // Send results back to master
        string results_str = vector_to_string(worker_results);
        send_string(results_str, 0);

        printf("Process %d processed chunk in %.6f seconds (found %lu matches)\n",
               rank, end_time - start_time, worker_results.size());
    }

    MPI_Finalize();
    return 0;
}