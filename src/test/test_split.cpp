#include "../utils/all.h"
#include <string>
#include <vector>
using namespace fms;
using namespace std;


int main() {
    string line = "hello world\tgo that\tthat go"; 
    vector<string> vs = std::move(split(line, "\t"));

    for(auto it=vs.begin(); it!=vs.end(); ++it) {
        cout << "line\t" << *it << endl;
    }

    return 0;
}
