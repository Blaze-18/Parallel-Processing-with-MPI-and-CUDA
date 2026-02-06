#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

// TO RUN THIS CODE: 
// Compile : mpic++ file_name.cpp -o out
// RUN: mpirun -n <No. of process> <text_file_name> <Search String>
// Example: mpirun -n 5 phonebook1.txt ANTU


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

    vector<char> buf(len);              // safer than new/delete
    MPI_Recv(buf.data(), len, MPI_CHAR, sender, 1, MPI_COMM_WORLD, &status);

    return string(buf.data());
}

// Converts a range of a vector of strings into one single string for transmission
string vector_to_string(const vector<string> &lines, int start, int end) {
    string result;
    for (int i = start; i < min((int)lines.size(), end); i++)
        result += lines[i] + "\n";
    return result;
}

// Splits a large received string back into a vector of strings
vector<string> string_to_vector(const string &text) {
    vector<string> lines;
    istringstream iss(text);
    string line;

    while (getline(iss, line))
        if (!line.empty()) lines.push_back(line);

    return lines;
}

// Reads raw lines from multiple files into a vector
void read_phonebook(const vector<string> &files, vector<string> &lines) {
    for (const string &file : files) {
        ifstream f(file);
        if (!f.is_open()) {
            cerr << "Could not open file: " << file << endl;
            continue;
        }

        string line;
        while (getline(f, line))
            if (!line.empty()) lines.push_back(line);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            cerr << "Usage: mpirun -n <procs> " << argv[0]
                 << " <file1>... <search_term>\n";
        MPI_Finalize();
        return 1;
    }

    string search_term = argv[argc - 1];
    double start_time, end_time;
    
    // MUST WRITE ON YOUR OWN ------------------ Start
    if (rank == 0) {
        // ===== MASTER =====

        // Collect filenames
        vector<string> files;
        for (int i = 1; i < argc - 1; i++)
            files.push_back(argv[i]);

        // Read all lines
        vector<string> all_lines;
        read_phonebook(files, all_lines);

        int total = all_lines.size();
        int chunk = (total + size - 1) / size;

        // Send chunks to workers
        for (int i = 1; i < size; i++)
            send_string(vector_to_string(all_lines, i * chunk, (i + 1) * chunk), i);

        start_time = MPI_Wtime();

        vector<string> final_matches;
        
        
        // Master searches its own chunk
        for (int i = 0; i < min(chunk, total); i++){
            if (all_lines[i].find(search_term) != string::npos){
                  final_matches.push_back(all_lines[i]);
                }
        }

        // Receive worker results
        for (int i = 1; i < size; i++) {
            vector<string> worker = string_to_vector(receive_string(i));
            final_matches.insert(final_matches.end(), worker.begin(), worker.end());
        }

        // Sort results
        sort(final_matches.begin(), final_matches.end());

        end_time = MPI_Wtime();

        // Write to file
        ofstream out("output.txt");
        for (const string &s : final_matches)
            out << s << "\n";

        cout << "Search complete. Found " << final_matches.size() << " matches." << endl;
        printf("Total execution time (including sort): %f seconds.\n", end_time - start_time);

    } else {
        // ===== WORKER =====

        vector<string> local_lines = string_to_vector(receive_string(0));

        start_time = MPI_Wtime();

        string local_matches;
        for (const string &line : local_lines)
            if (line.find(search_term) != string::npos)
                local_matches += line + "\n";

        end_time = MPI_Wtime();

        send_string(local_matches, 0);

        printf("Process %d processed %lu lines in %f seconds.\n",
               rank, local_lines.size(), end_time - start_time);
    }
    // MUST WRITE ON YOUR OWN -----------------------------FINISH

    MPI_Finalize();
    return 0;
}
