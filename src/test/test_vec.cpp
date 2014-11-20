#include "../utils/all.h"
using namespace fms;

int main()
{
    Vec v1(10);
    v1.init(10, true);

    Vec v2(10);
    v2.init(10, true);

    v1.display();
    v2.display();

    return 0;
}
