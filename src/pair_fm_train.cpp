#include "fms/pair_fm.h"

int main(int argc, char *argv[]) 
{
    using namespace fms;
    using namespace std;
    std::string train_path{""}, test_path{""}; 
    int dim = 10;
    index_t batch_size = 10000;
    int num_iters = 10;
    int num_threads = 3;
    string input_format = "";
    double learning_rate = 0.1;
    std::string output_model_path{""};
    // set up paramter engine
    CMDLine cmdline(argc, argv);
    string param_train_path = cmdline.registerParameter("train_data", "path of training set");
    string param_test_path = cmdline.registerParameter("test_data", "path of test set");
    string param_dim = cmdline.registerParameter("dim", "dimention of fm vector");
    string param_learning_rate = cmdline.registerParameter("learning_rate", "learning rate of ADAGRAD");
    string param_num_iters = cmdline.registerParameter("num_iters", "number of iterations");
    string param_num_threads = cmdline.registerParameter("num_threads", "number of threads");
    string param_batch_size = cmdline.registerParameter("batch_size", "size of a mini-batch");
    string param_output_model_path = cmdline.registerParameter("output_model_path", "path to output the model");
    string param_input_format = cmdline.registerParameter("input_format", "format of the dataset: Data/IDData");
    string param_help = cmdline.registerParameter("help", "this screen");
    // parse parameters
    if(cmdline.hasParameter(param_help) || argc == 1) {
        cout << endl;
        cout << "===================================================================" << endl;
        cout << "list -> pair -> FM -> ranknet as loss function" << endl;
        cout << "===================================================================" << endl;
        cmdline.print_help();
        cout << endl;
        cout << endl;
        return 0;
    }
    if(!cmdline.hasParameter(param_train_path)) {
        LOG(ERROR) << "missing parameter: train_path";
	return 0;
    }
    train_path = cmdline.getValue(param_train_path);
    if(cmdline.hasParameter(param_test_path)) {
        test_path = cmdline.getValue(param_test_path);
    }
    if(cmdline.hasParameter(param_dim)) {
        dim = std::atoi( cmdline.getValue(param_dim).c_str());
    }
    if(cmdline.hasParameter(param_learning_rate)) {
        param_learning_rate = std::atof(cmdline.getValue(param_learning_rate).c_str());
    }
    if(cmdline.hasParameter(param_num_iters)) {
        num_iters = std::atoi(cmdline.getValue(param_num_iters).c_str());
    }
    if(cmdline.hasParameter(param_num_threads)) {
        num_threads = std::atoi(cmdline.getValue(param_num_threads).c_str());
    }
    if(cmdline.hasParameter(param_batch_size)) {
        batch_size = std::atoi(cmdline.getValue(param_batch_size).c_str());
    }
    if(cmdline.hasParameter(param_output_model_path)) {
        output_model_path = cmdline.getValue(param_output_model_path); 
    }
    if(!cmdline.hasParameter(param_input_format)) {
        LOG(ERROR) << "missing parameter input_format";
        return 0;
    } else {
        input_format = cmdline.getValue(param_input_format);
    }
    LOG(INFO) << "get parameter\t" << "train_path:\t" << train_path;
    LOG(INFO) << "get parameter\t" << "test_path:\t" << test_path;
    LOG(INFO) << "get parameter\t" << "dim:\t" << dim;
    LOG(INFO) << "get parameter\t" << "batch_size:\t" << batch_size;
    LOG(INFO) << "get parameter\t" << "num_iters:\t" << num_iters;
    LOG(INFO) << "get parameter\t" << "num_threads:\t" << num_threads;
    LOG(INFO) << "get parameter\t" << "learning_rate:\t" << learning_rate;
    LOG(INFO) << "get parameter\t" << "input_format:\t" << input_format;
    LOG(INFO) << "get parameter\t" << "output_model_path:\t" << output_model_path;
    // run model
    if(input_format == "Data") {
        PairFM<ListData, ListInstance> fms(train_path, test_path, dim, batch_size, learning_rate);
        fms.train(num_threads, num_iters);
        if(!output_model_path.empty()) {
            fms.model_to(output_model_path);
        }
    } 
    if(input_format == "IDData") {
        PairFM<ListData, ListInstance> fms(train_path, test_path, dim, batch_size, learning_rate);
        fms.train(num_threads, num_iters);
        if(!output_model_path.empty()) {
            fms.model_to(output_model_path);
        }
    }
    return 0;
}


