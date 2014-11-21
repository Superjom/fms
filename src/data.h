#ifndef _FMS_DATA_H_
#define _FMS_DATA_H_
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <limits>
#include "utils/all.h"
#include "core/common.h"
namespace fms {
using std::cin;
using std::cout;
using std::endl;

template <typename Inst=Instance>
class BaseData {
public:
    explicit BaseData() { }
    void set_path(const std::string &path) {
        CHECK(!path.empty());
        LOG(WARNING) << "loading data from " << path; 
        _path = path;
        // 直接读取记录
        std::ifstream infile(path.c_str());
        CHECK(infile) << "can not open file:\t" << path;
        std::string line;
        while(std::getline(infile, line)) {
            CHECK(line.size() > 0) << "error parse empty line:\r" << line;
            Inst ins;
            parse_instance(line.c_str(), ins);
            _instances.push_back(std::move(ins));
        }
        infile.close();
        LOG(INFO) << "data.size:\t" << size();
        LOG(INFO) << "data.max_key:\t" << max_key();
        LOG(INFO) << "data.min_val:\t" << min_val();
        LOG(INFO) << "data.max_val:\t" << max_val();
    }
    virtual void parse_instance(const char *line, Inst &ins) = 0;
    std::string path() const {
        return _path;
    }
    index_t max_key() const {
        return _max_key;
    }
    double max_val() const {
        return _max_val;
    }
    double min_val() const {
        return _min_val;
    }
    const std::vector<Inst>& instances() const {
        return _instances;
    }
    std::vector<Inst>& instances() {
        return _instances;
    }
    index_t size() const {
        return _instances.size();
    }
protected:
    std::string _path;
    std::vector<Inst> _instances;
    index_t _max_key{0};
    double _min_val = std::numeric_limits<double>::max();
    double _max_val = -_min_val;
}; // class Data


class Data : public BaseData<Instance> {
public:
    explicit Data() { }
    /*
     * 格式： 空格隔开 key必须递增
     * 0.34 12:0.1 23:0.5 34:0.7
     */
    void parse_instance(const char *line, Instance &ins) {
        char *cursor;
        CHECK((ins.target=strtod(line, &cursor), cursor!=line)) << "target parse error!";
	    ins.feas.clear();
        line = cursor;
        // 忽略开头空格
        while( *(line + count_spaces(line)) != 0) {
            index_t key;
            double value;
            CHECK((key = (uint64_t)strtoull(line, &cursor, 10), cursor!=line));
            line = cursor;
            CHECK(*line++ == ':');
            CHECK((value = (double)strtod(line, &cursor), cursor!=line));
            line = cursor;
	        ins.feas.push_back( Instance::Item(key, value));
            _max_val = std::max(_max_val, value);
            _min_val = std::min(_min_val, value);
            _max_key = std::max(_max_key, key);
        }
        //ins.n_feas = ins.feas.size();
    }
}; // class Data

// 输入如下格式： 只有ID
// 0.7 331 3444
class IDData : public BaseData<Instance> {
public:
    explicit IDData() { }
    /*
     * 格式：
     * target: id id id
     */
    void parse_instance(const char *line, Instance &ins) {
        char *cursor;
        CHECK((ins.target=strtod(line, &cursor), cursor!=line)) << "target parse error!";
	    ins.feas.clear();
        line = cursor;
        // 忽略开头空格
        while( *(line + count_spaces(line)) != 0) {
            index_t key;
            double value = 1.0;
            CHECK((key = (uint64_t)strtoull(line, &cursor, 10), cursor!=line));
            line = cursor;
            //CHECK(*line++ == ':');
            //CHECK((value = (double)strtod(line, &cursor), cursor!=line));
            //line = cursor;
	        ins.feas.push_back( Instance::Item(key, value));
            //_max_val = std::max(_max_val, value);
            //_min_val = std::min(_min_val, value);
            _max_key = std::max(_max_key, key);
        }
        //ins.n_feas = ins.feas.size();
        _min_val = 0.0;
        _max_val = 1.0;
    }
};


class ListData : public BaseData<ListInstance> {
public:
    explicit ListData() { }
    // 格式： 空格隔开 key必须递增
    // 0.34 12:0.1 23:0.5 34:0.7
    void parse_instance(const char *line, ListInstance &list_ins) {
        std::vector<std::string> cols = fms::split(std::string(line), "\t");
        CHECK(!cols.empty());
        // 第一个 field 是 共同前缀 prefix
        std::string &head = cols[0];
        list_ins.prefix = std::move(parse_items(head.c_str()));
        std::vector<Instance> list;
        for(index_t i = 1; i < cols.size(); i++) {
            Instance ins;
            const char* line = cols[i].c_str();
            char *cursor;
            CHECK((ins.target = std::strtof(line, &cursor), cursor!=line));
            line = cursor;
            ins.feas = std::move(parse_items(line));
            //ins.n_feas = ins.feas.size();
            list_ins.list.push_back(std::move(ins));
        }
    }
    std::vector<Instance::Item> parse_items(const char *line) {
        //line = trim(std::string(line)).c_str();
        //cout << "line:\t" << line << endl;
        std::vector<Instance::Item> tmp;
        while( *(line + count_spaces(line)) != 0) {
            index_t key;
            double value;
            char *cursor;
            CHECK((key = (uint64_t)strtoull(line, &cursor, 10), cursor!=line));
            line = cursor;
            CHECK(*line++ == ':');
            CHECK((value = (double)strtod(line, &cursor), cursor!=line));
            line = cursor;
            //cout << "key:value\t" << key << ":" << value << endl;
            tmp.push_back( Instance::Item(key, value));
            _max_val = std::max(_max_val, value);
            _min_val = std::min(_min_val, value);
            _max_key = std::max(_max_key, key);
        }
        return std::move(tmp);
    }
}; // end class ListData


}; // end namespace fms

#endif
