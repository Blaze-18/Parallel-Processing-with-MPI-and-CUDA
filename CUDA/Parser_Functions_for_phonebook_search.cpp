// USAGE: In the main code just sweap the parsePhonebook function with the one that supports your desired format

// Accepts format: Name,contact_number
// Example: Anan,0123123123
//          Nirjhor,012301203
// Multiple names is not supported: Example: Shahriar Anan,0123123123 (NOT SUPPORTED!!!!!!!!)
bool parsePhonebook(const string& file_name,
                    vector<string>& names,
                    vector<string>& numbers) {

    ifstream file(file_name);
    if (!file.is_open()) return false;

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;

        int pos = line.find(",");
        if (pos == string::npos) continue;

        names.push_back(line.substr(0, pos));
        numbers.push_back(line.substr(pos + 1));
    }

    return !names.empty();
}

// Accepts format: Name contact_number
// Example: Anan 0123123123
//          Nirjhor 012301203
// Multiple names is not supported: Example: Shahriar Anan 0123123123 (NOT SUPPORTED!!!!!!!!)
bool parsePhonebook(const string& file_name,
                    vector<string>& names,
                    vector<string>& numbers) {

    ifstream file(file_name);
    if (!file.is_open()) return false;

    string name, number;
    while (file >> name >> number) {
        names.push_back(name);
        numbers.push_back(number);
    }

    return !names.empty();
}

// Accepts format: "Name","contact_number"
// Example: "Anan","0123123123"
//          "Nirjhor","012301203"
// Multiple names is supported: Example: "Shahriar Anan","0123123123" (YESSSSS SUPPORTED!!!!!!!!)
bool parsePhonebook(const string& file_name,
                    vector<string>& names,
                    vector<string>& numbers) {

    ifstream file(file_name);
    if (!file.is_open()) return false;

    string line;
    while (getline(file, line)) {
        if (line.size() < 5) continue;

        int pos = line.find("\",\"");
        if (pos == string::npos) continue;

        names.push_back(line.substr(1, pos - 1));
        numbers.push_back(line.substr(pos + 3, line.size() - pos - 4));
    }

    return !names.empty();
}

// Accepts format: Name-contact_no
// Example: Anan-0123123123
//          Nirjhor-012301203
// Multiple names is not supported: Example: Shahriar Anan-0123123123 (NOT SUPPORTED!!!!!!!!)
bool parsePhonebook(const string& file_name,
                    vector<string>& names,
                    vector<string>& numbers) {

    ifstream file(file_name);
    if (!file.is_open()) return false;

    string line;
    while (getline(file, line)) {
        int pos = line.find("-");
        if (pos == string::npos) continue;

        names.push_back(line.substr(0, pos));
        numbers.push_back(line.substr(pos + 1));
    }

    return !names.empty();
}

// Accepts format: "id:","Id_no","Name","contact_number"
// Example: "id:","01","Anan","0123123123"
//          "id:","02","Nirjhor","012301203"
// Multiple names is not supported: Example: "id:","01","Shahriar Anan","0123123123"(YESSSS SUPPORTED!!!!!!!!)
bool parsePhonebook(const string& file_name,
                    vector<string>& names,
                    vector<string>& numbers) {

    ifstream file(file_name);
    if (!file.is_open()) return false;

    string line;
    while (getline(file, line)) {
        vector<string> parts;
        string temp;

        for (char c : line) {
            if (c == '"') continue;
            if (c == ',') {
                parts.push_back(temp);
                temp.clear();
            } else {
                temp += c;
            }
        }
        parts.push_back(temp);

        if (parts.size() >= 4) {
            names.push_back(parts[2]);
            numbers.push_back(parts[3]);
        }
    }

    return !names.empty();
}

// Accepts format: "id: Id_no","Name","contact_no"
//Example: "id: 25","John Michael Doe","01700000000"
bool parsePhonebook(const string& file_name,
                    vector<string>& names,
                    vector<string>& numbers) {

    ifstream file(file_name);
    if (!file.is_open()) return false;

    string line;
    while (getline(file, line)) {

        vector<string> parts;
        string temp;
        bool inside_quote = false;

        for (char c : line) {
            if (c == '"') {
                inside_quote = !inside_quote;
                continue;
            }

            if (c == ',' && !inside_quote) {
                parts.push_back(temp);
                temp.clear();
            } else {
                temp += c;
            }
        }
        parts.push_back(temp);

        // Expected: [id: Id_no] , [Name] , [contact_no]
        if (parts.size() >= 3) {
            names.push_back(parts[1]);
            numbers.push_back(parts[2]);
        }
    }

    return !names.empty();
}

