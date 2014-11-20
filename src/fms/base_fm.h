#ifndef _FMS_BASE_FM_H_
#define _FMS_BASE_FM_H_
#include "../data.h"
#include "../utils/all.h"
#include "../core/all.h"
#include <thread>
#include <atomic>
#include <mutex>

namespace fms {

// 基本训练的多线程版本的框架
template<typename SGD=KLdistSGD, typename FMParam=AdaGradFMParam, typename DataType=IDData, typename InsType=Instance>
class BaseFM : public VirtualObject {
public:
    explicit BaseFM(const std::string &train_path, const std::string &test_path, int dim, index_t batch_size, double learning_rate) :\
        _train_path(train_path), \
        _test_path(test_path),
        _batch_size(batch_size), \
        _dim(dim)
    {
        CHECK(!train_path.empty());
        LOG(INFO) << "load trainset";
        _train_data.set_path(train_path);
        if(!_test_path.empty()) {
            LOG(INFO) << "load testset" ;
            _test_data.set_path(test_path);
        }
        int num_feas = _train_data.max_key() + 1;
        _fm.init_feature(dim, num_feas);
        _fm.init(dim, num_feas, learning_rate);
        _sgd.set_fm(_fm);
        _train_data_size = _train_data.size();
        _test_data_size = _test_data.size();
    }
    void train(int num_threads, int num_iters) {
        _batch_size = std::min(_batch_size, _train_data_size);
        index_t num_batches = _train_data_size / _batch_size;
        if (num_batches * _batch_size < _train_data_size) num_batches++;
        index_t span = _batch_size / num_threads;

        LOG(INFO) << "split batch number:\t" << num_batches;
        LOG(INFO) << "eatch thread learns\t" << span << "\t records";
        for (int iter = 0; iter < num_iters; iter++) {
            double train_cost = std::move(train_iter(num_threads));
            double test_cost = std::move(test_iter(num_threads));
            if(_test_path.empty()) {
                LOG(INFO) << "iter\t" << iter << "\tcost\t" << train_cost << "\tNaN num\t" << _nan_num;
            } else {
                LOG(INFO) << "iter\t" << iter << "\ttrain\t" << train_cost << "\ttest\t" << test_cost << "\tNaN num\t" << _nan_num;
            }
        }
    }
    void model_to(const std::string &path) {
        LOG(INFO) << "output model to " << path;
        _fm.model_to(path);
    }
protected:
    double train_iter(int num_threads) {
        index_t num_batches = _train_data_size / _batch_size;
        if (num_batches * _batch_size < _train_data_size) num_batches++;
        index_t span = _batch_size / num_threads;
        InsType* instances = &_train_data.instances()[0];
        _nan_num = 0;   // nan_num 清 0
        for (int i_batch = 0; i_batch < num_batches; ++i_batch) {
            std::vector<std::thread> threads;
            for(int i = 0; i < num_threads; ++i) {
                index_t start = i * span + i_batch * _batch_size;
                index_t end = std::min(_train_data_size-1, span * (i + 1) + i_batch * _batch_size );
                //LOG(INFO) << "start:end\t" << start << "\t" << end;
                // 每个进程分配一个范围的数据
                std::thread t1([this](InsType* instances, index_t start, index_t end) {
                    double cost;
                    for(index_t i = start; i < end; ++i ) {
                        const InsType &ins = instances[i];
                        cost = _sgd.learn_instance(ins);
                        if(std::isnan(cost)) {
                            //LOG(WARNING) << "nan detected"; 
                            _nan_num ++;
                            continue;
                        }
                        _train_cost.cumulate(cost);
                    }
                }, instances, start, end);
                threads.push_back(std::move(t1));
            } // end for num_threads
            for(auto &t : threads) {
                t.join();
            }  // end join threads
            _fm.batch_push();
        } //  end batch
        double cost = _train_cost.norm();
        _train_cost.reset();
        return cost;
    } // end train
    double test_iter(int num_threads) {
        if(_test_path.empty()) return 0.0;
        CHECK(_test_data_size > 0);
        index_t span = _test_data_size / num_threads;
        InsType* instances = &_test_data.instances()[0];
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; i++) {
            index_t start = span * i;
            index_t end = std::max(_test_data_size, span * (i + 1));

            std::thread t1([this](InsType* instances, index_t start, index_t end) {
                for(index_t i = start; i < end; ++i) {
                    double cost;
                    const InsType &ins = instances[i];
                    cost = _sgd.forward(ins);
                    _test_cost.cumulate(cost);
                }
            }, instances, start, end);
            threads.push_back(std::move(t1));
        } // end for
        for(auto &t : threads) {
            t.join();
        }
        double cost = _test_cost.norm();
        _test_cost.reset();
        return cost;
    }

private:
    std::string _train_path;
    std::string _test_path;
    std::atomic<index_t> _cur_id{0};        // 当前扫描的data id
    std::atomic<index_t> _cur_iter_no{0} ;   // 当前训练的轮数
    std::atomic<index_t> _nan_no{0};
    index_t _train_data_size = 0;     // 训练集的记录数目
    index_t _test_data_size = 0;
    index_t _batch_size;
    int _dim;
    FMParam _fm;
    SGD _sgd;
    DataType _train_data;
    DataType _test_data;
    Cost _train_cost;
    Cost _test_cost;
    index_t _nan_num{0};
    std::mutex _mutex;
}; // end class

};  // end namespace fms
#endif

