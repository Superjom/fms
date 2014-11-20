#include "../data.h"

using namespace fms;
using namespace std;


int main() 
{
    Data data;
    data.set_path("1.txt");
    cout << "size:\t" << data.size() << endl;
    cout << "max_key:\t" << data.max_key() << endl;
    cout << "min_val:\t" << data.min_val() << endl;
    cout << "max_val:\t" << data.max_val() << endl;
}
