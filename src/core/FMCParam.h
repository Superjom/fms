#ifndef _FMS_CORE_FMCPARAM_H_
#define _FMS_CORE_FMCPARAM_H_
#include <random>
#include <sstream>
#include "../utils/all.h"
#include "./common.h"
#include "./VirtualObject.h"

namespace fms {

// 标记输出输入模型的版本格式
// 只能输入同版本的模型
const int OUTPUT_MODEL_VERSION = 0;

struct FMValue {
    // AdaGrad parameters
    double lr2sum = 0.0;
    fms::Vec fm2sum; // need init
    // model paramters
    double lr_w = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) ;
    fms::Vec fm_w;
    double n = 0.0;  // counter

    FMValue(int _dim) {
        fm2sum.init(_dim);
        fm_w.init(_dim, true);
    }
    friend std::ostream& operator<< (std::ostream& os, const FMValue& other) {
        os << "lr_w\t" << other.lr_w << endl;
        os << "lr2sum\t" << other.lr2sum << endl;
        os << "fm_w\t" << other.fm_w << endl;
        os << "fm2sum\t" << other.fm2sum << endl;
        os << "n\t" << other.n << endl;
        return os;
    }
    /*
     * 输出格式：
     * lr_w \t lr2sum \t fm_w \t fm2sum
     */
    virtual std::string to_str() const {   // 用于输出模型
        std::stringstream ss;
        ss << lr_w << "\t" << lr2sum << "\t";
        for (auto it = fm_w.begin(); it != fm_w.end(); ++it) {
            if(it != fm_w.end() - 1) {
                ss << *it << " ";
            } else {
                ss << *it;
            }
        }
        ss << "\t";
        for (auto it = fm2sum.begin(); it != fm2sum.end(); ++it) {
            if(it != fm2sum.end() - 1) {
                ss << *it << " ";
            } else {
                ss << *it;
            }
        }
        return std::move(ss.str());
    }
    void from_str(const char *line, int dim) {
        char *cursor;
        CHECK((lr_w=strtod(line, &cursor), cursor!=line)) << "lr_w parse error!";
        line = cursor;
        CHECK((lr2sum=strtod(line, &cursor), cursor!=line)) << "lr2sum parse error!";
        line = cursor;
        std::vector<double> vec;
        // parse fm_w
        for(int i=0; i < dim; i++) {
            double value{0.0};
            CHECK((value = strtod(line, &cursor), cursor!=line)) << "fm_w parse error!";
            line = cursor;
            vec.push_back(value);
        }
        fm_w = vec;
        CHECK(fm_w.size() > 0);
        // parse fm2sum
        vec.clear();
        //CHECK(line[0] == '\t') << "tab parse error";
        //line++;
        for(int i=0; i < dim; i++) {
            double value{0.0};
            CHECK((value = strtod(line, &cursor), cursor!=line)) << "fm2sum parse error!";
            line = cursor;
            vec.push_back(value);
        }
        fm2sum = vec;
        CHECK(fm_w.size() == fm2sum.size()) << fm_w.size() << "\t" << fm2sum.size();
    }
};
/* 
 * 基本的FM特征定义
 */
class FMParam : public VirtualObject {
public:
    explicit FMParam() { }
    explicit FMParam(int dim, index_t num_feas) : _dim(dim), _num_feas(num_feas) {
        CHECK(_dim >= 0);
        CHECK(_num_feas > 0);
        init_feature(dim, num_feas);
    }
    void init_feature(int dim, index_t num_feas) {
        _num_feas = num_feas;
        _dim = dim;
        for(index_t i = 0; i < num_feas; i++) {
            FMValue tmp(dim);
            _feas.push_back(std::move(tmp));
        }
    }
    const FMValue &feature(index_t key) const {
        CHECK((key>=0 && key<_num_feas)) << key;
        return _feas[key];
    }
    FMValue &feature(index_t key) {
        CHECK((key>=0 && key<_num_feas)) << key;
        return _feas[key];
    }
    int dim() const {
        return _dim;
    }
    index_t num_feas() const {
        return _num_feas;
    }
    friend std::ostream & operator<< (std::ostream &os, const FMParam &other) {
        for(auto it = other._feas.begin(); it != other._feas.end(); ++it) {
            os << *it << endl;
        }
        return os;
    }
    void model_to(const std::string &path) {
        LOG(WARNING) << "output model to\t " << path;
        CHECK(!path.empty());
        std::ofstream file(path);
        // 输出元信息
        file << "#VERSION\t" << OUTPUT_MODEL_VERSION << endl;
        file << "#dim\t" << _dim << endl;
        file << "#num_features\t" << _num_feas << endl;
        for(index_t i = 0; i < _feas.size(); ++i) {
            FMValue &fv = _feas[i];
            file << i << "\t" << fv.to_str() << endl;
        }
    }
    void from_model(const std::string &path) {
        LOG(WARNING) << "load model from\t " << path;
        std::ifstream file(path);
        std::string line;
        std::string head, value;
        std::vector<string> cols;
        std::stringstream ss;
        // 读取元信息
        file >> head >> value;
        CHECK(head == "#VERSION");
        CHECK(std::atoi(value.c_str()) == OUTPUT_MODEL_VERSION);
        file >> head >> value;
        CHECK(head == "#dim") << head;
        _dim = std::atoi(value.c_str());
        file >> head >> value;
        CHECK(head == "#num_features");
        _num_feas = std::atoi(value.c_str());
        LOG(INFO) << "version:\t" << OUTPUT_MODEL_VERSION;
        LOG(INFO) << "dim:\t" << _dim;
        LOG(INFO) << "num_features:\t" << _num_feas;
        _feas.clear();
        getline(file, line);
        while(getline(file, line)) {
            std::size_t pos = line.find('\t')+1;
            std::string cline = std::move(line.substr(pos));
            FMValue fv(_dim);
            fv.from_str(cline.c_str(), _dim);
            _feas.push_back(std::move(fv));
        }
        LOG(INFO) << "load features:\t" << _feas.size();
    }

protected:
    int _dim = 0;
    index_t _num_feas = 0;
private:
    std::vector<FMValue> _feas;
};

}; // end namespace fms


#endif
