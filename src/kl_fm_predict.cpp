#include <sstream>
#include "fms/kl_fm.h"

namespace fms {

extern const int OUTPUT_MODEL_VERSION;

template<typename SGD=KLdistSGD, typename FMParam=AdaGradFMParam, typename DataType=Data>
class KLPredictor : public VirtualObject {
public:
    explicit KLPredictor(const std::string &data_path, 
        const std::string &model_path, 
        const std::string &out_path) :\
        _data_path(data_path), \
        _model_path(model_path),
        _out_path(out_path)
    {
        CHECK(!_data_path.empty());
        CHECK(!_model_path.empty());
        CHECK(!_out_path.empty());
        _data.set_path(_data_path);
        LOG(INFO) << "load model from\t" << _model_path;
        _fm.from_model(_model_path);
        _sgd.set_fm(_fm);
    }
    // 预测并输出预测值
    void predict_out() {
        Cost _cost;
        ofstream file(_out_path);
        const Instance *instances = &_data.instances()[0];
        for(index_t i = 0; i < _data.size(); ++i) {
            double cost;
            const Instance &ins = instances[i];
            Vec x_v_sum(_fm.dim());
            double q, y;
            cost = _sgd.forward(ins, x_v_sum, q, y);
            // 溢出测试
            if(std::isnan(cost)) {
                // TODO set q to mean value
                q = (_data.max_val() + _data.min_val()) / 2.0;
            }
            cost = KLdistance(ins.target, q);
            _cost.cumulate(cost);
            file << q << endl;
        }
        file.close();
        LOG(INFO) << "cost:\t" << _cost.norm();
    }
private:
    std::string _data_path;
    std::string _model_path;
    std::string _out_path;
    int _dim;
    index_t _num_feas{0};
    FMParam _fm;
    SGD _sgd;
    DataType _data;
};  // end class KLPredictor


}; // end namespace fms



int main(int argc, char *argv[]) 
{
    using namespace std;
    using namespace fms;

    string data_path{""};
    string model_path{""}; // 载入模型文件路径
    string out_path{""};   // 输出结果的路径
    string input_format = "";

    CMDLine cmdline(argc, argv);
    string param_data_path = cmdline.registerParameter("data_path", "path of the data to predict");
    string param_model_path = cmdline.registerParameter("model_path", "path of the model to load");
    string param_out_path = cmdline.registerParameter("out_path", "path of the predicted values to output");
    string param_help = cmdline.registerParameter("help", "this screen");
    string param_input_format = cmdline.registerParameter("input_format", "format of the dataset: Data/IDData");

    if(cmdline.hasParameter(param_help) || argc == 1) {
        cout << endl;
        cout << "===================================================================" << endl;
        cout << "FM -> sigmoid(map to probability) -> KL-distance as loss function " << endl;
        cout << "===================================================================" << endl;
        cmdline.print_help();
        cout << endl;
        cout << endl;
        return 0;
    }
    if(!cmdline.hasParameter(param_data_path) || \
        !cmdline.hasParameter(param_model_path) || \
        !cmdline.hasParameter(param_out_path)) {
        LOG(ERROR) <<"missing parameters: data_path or model_path or out_path";
        return 0;
    }
    if(!cmdline.hasParameter(param_input_format)) {
        LOG(ERROR) << "missing parameter input_format";
        return 0;
    } else {
        input_format = cmdline.getValue(param_input_format);
    }

    data_path = cmdline.getValue(param_data_path);
    model_path = cmdline.getValue(param_model_path);
    out_path = cmdline.getValue(param_out_path);
    if(input_format == "Data") {
        KLPredictor<KLdistSGD, AdaGradFMParam, Data> predictor (data_path, model_path, out_path);
        predictor.predict_out();
    }
    if(input_format == "IDData") {
        KLPredictor<KLdistSGD, AdaGradFMParam, IDData> predictor (data_path, model_path, out_path);
        predictor.predict_out();
    }

    return 0;
}
