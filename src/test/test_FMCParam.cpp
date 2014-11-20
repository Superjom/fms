#include "../core/FMCParam.h"
using namespace fms;

int main()
{
    AdaGradFMParam param(5, 3);
    cout << param;

    std::vector<double> vec1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> vec2 = {1.1, 2.1, 3.1, 4.1, 5.1};
    SGDGradValue grad1(1.0, vec1);
    SGDGradValue grad2(2.0, vec2);
    cout << "grad1:\t" << grad1;
    cout << "grad2:\t" << grad2;
    cout << "after merge" << endl;
    grad1.merge_with(grad2);
    grad1.merge_with(grad2);
    cout << "grad1:\t" << grad1;

    cout << "feature 2:\t" << param.feature(2);
    cout << "before commit" << endl;
    param.batch_commit(2, grad1);
    cout << "feature 2:\t" << param.feature(2);
    param.batch_commit(0, grad2);
    param.batch_push();
    cout << "feature 2:\t" << param.feature(2);





};
