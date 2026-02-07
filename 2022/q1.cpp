
/*
    Question: 1. Write a program using MPI to count the words in a file and sort it in descending order of 
    frequency of words i.e., highest occurring words must come first and the least occurring word must come last
    
    Input: No. of processes, (Text input from file) 
    Output: Total processing time, top 10 occurrences of string

*/

/*
    To Run this program
    First make sure you are inside the directory
    Compile: mpic++ q1.cpp -o wordcount
    Run: mpirun -n <number_of_processes> ./wordcount <input_file.txt>
    
    Example Run with 4 process:
    mpirun -n 4 ./wordcount text.txt
    

*/

#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

// Convert string to lowercase and remove punctuation
string clean_word(string w) {
    string res = "";
    for (char c : w) {
        if (isalpha(c))
            res += tolower(c);
    }
    return res;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            cout << "Usage: mpirun -n <processes> ./wordcount <file.txt>\n";
        MPI_Finalize();
        return 0;
    }

    double start_time = MPI_Wtime();

    vector<string> all_words;
    int total_words = 0;

    // Root reads file
    if (rank == 0) {
        ifstream file(argv[1]);
        string word;

        while (file >> word) {
            word = clean_word(word);
            if (!word.empty())
                all_words.push_back(word);
        }

        total_words = all_words.size();
    }

    // Broadcast total word count
    MPI_Bcast(&total_words, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine chunk for each process
    int chunk = total_words / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? total_words : start + chunk;

    // Scatter words manually (simple approach)
    vector<string> local_words;

    if (rank == 0) {
        for (int i = start; i < end; i++)
            local_words.push_back(all_words[i]);

        // send to other processes
        for (int p = 1; p < size; p++) {
            int s = p * chunk;
            int e = (p == size - 1) ? total_words : s + chunk;
            int count = e - s;

            MPI_Send(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD);

            for (int i = s; i < e; i++) {
                int len = all_words[i].size();
                MPI_Send(&len, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
                MPI_Send(all_words[i].c_str(), len, MPI_CHAR, p, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        int count;
        MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < count; i++) {
            int len;
            MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            char* buf = new char[len + 1];
            MPI_Recv(buf, len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            buf[len] = '\0';

            local_words.push_back(string(buf));
            delete[] buf;
        }
    }

    // Local word frequency
    unordered_map<string, int> local_freq;
    for (auto& w : local_words)
        local_freq[w]++;

    // Gather all maps to root
    if (rank == 0) {
        unordered_map<string, int> global_freq = local_freq;

        for (int p = 1; p < size; p++) {
            int map_size;
            MPI_Recv(&map_size, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < map_size; i++) {
                int len, count;
                MPI_Recv(&len, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                char* buf = new char[len + 1];
                MPI_Recv(buf, len, MPI_CHAR, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                buf[len] = '\0';

                MPI_Recv(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                global_freq[string(buf)] += count;
                delete[] buf;
            }
        }

        // Sort by frequency descending
        vector<pair<string, int>> sorted(global_freq.begin(), global_freq.end());
        sort(sorted.begin(), sorted.end(),
             [](auto& a, auto& b) { return a.second > b.second; });

        double end_time = MPI_Wtime();

        cout << "\nTop 10 most frequent words:\n";
        for (int i = 0; i < min(10, (int)sorted.size()); i++)
            cout << sorted[i].first << " : " << sorted[i].second << "\n";

        cout << "\nTotal Processing Time: " << (end_time - start_time) << " seconds\n";
    } else {
        int map_size = local_freq.size();
        MPI_Send(&map_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        for (auto& [word, count] : local_freq) {
            int len = word.size();
            MPI_Send(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(word.c_str(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
