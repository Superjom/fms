#ifndef _FMS_UTILS_VEC_H_
#define _FMS_UTILS_VEC_H_
#include "common.h"
#include <cmath>

namespace fms {

using namespace std;
class Vec {
public:
    typedef double value_type;
    typedef uint32_t index_t;
	typedef vector<value_type>::iterator iterator;
    typedef vector<value_type>::const_iterator const_iterator;
	Vec() {}
	Vec(int size) {
		vec.reserve(size);
		vec.resize(size, 0.0);
	}
	Vec(const Vec &vec): vec(vec.vec) { }
	Vec(const vector<value_type> &vec) {
		this->vec = vec;
	}
    void init(int size, bool random_init=false) {
		vec.reserve(size);
		vec.resize(size, 0.0);
        if(random_init) randInit(0.5);
    }
	// operators ------------------------------------
	value_type dot(Vec &other) {
		value_type sum = 0.0;
		for(index_t i=0; i<other.size(); ++i) {
			sum += vec[i] * other[i];
		}
        return sum;
	}
	value_type mean() const {
		value_type res = 0.0;
		for(uint32_t i=0; i<vec.size(); i++) {
			res += vec[i];
		}
		return res / vec.size();
	}
	value_type sum() const {
		value_type res = 0.0;
		for(uint32_t i=0; i<vec.size(); i++) {
			res += vec[i];
		}
		return res;
	}
	friend Vec operator+(const Vec &other, value_type v) {
		Vec newVec(other);
		for(iterator it=newVec.begin(); it!=newVec.end(); ++it) {
			*it += v;
		}
		return newVec;
	}
	friend Vec& operator+=(Vec &a, const Vec &b) {
        CHECK(a.size() == b.size()) << a.size() << "\t" << b.size();
		for(index_t i=0; i<a.size(); ++i) {
			a[i] += b[i];
		}
		return a;
	}
	friend Vec& operator-=(Vec &a, const Vec &b) {
		for(index_t i=0; i<a.size(); ++i) {
			a[i] -= b[i];
		}
		return a;
	}
	friend Vec operator*(const Vec &other, value_type v) {
		Vec newVec(other);
		for(iterator it=newVec.begin(); it!=newVec.end(); ++it) {
			*it *= v;
		}
		return newVec;
	}
	friend Vec operator*(value_type v, const Vec &other) {
		Vec newVec(other);
		for(iterator it=newVec.begin(); it!=newVec.end(); ++it) {
			*it *= v;
		}
		return newVec;
	}
	friend Vec operator+(const Vec &a, const Vec &b) {
		CHECK(a.size() == b.size());
		Vec newVec(a);
		for(index_t i=0; i<a.size(); i++) {
			newVec[i] += b[i];
		}
		return newVec;
	}
	friend Vec operator-(const Vec &a, const Vec &b) {
		CHECK(a.size() == b.size());
		Vec newVec(a);
		for(index_t i=0; i<a.size(); i++) {
			newVec[i] -= b[i];
		}
		return newVec;
	}
	friend Vec operator*(const Vec &a, const Vec &b) {
		CHECK(a.size() == b.size());
		Vec newVec(a);
		for(index_t i=0; i<newVec.size(); i++) {
			newVec[i] *= b[i];
		}
		return newVec;
	}
	friend Vec& operator/= (Vec &a, value_type v) {
		for(iterator it=a.begin(); it!=a.end(); ++it) {
			*it /= v;
		}
        return a;
	}
	friend Vec operator/ (const Vec &a, value_type v) {
        Vec newVec(a);
		for(iterator it=newVec.begin(); it!=newVec.end(); ++it) {
			*it /= v;
		}
        return newVec;
	}
    friend Vec operator/ (const Vec &a, const Vec &b) {
        Vec newVec(a);
        for( index_t i = 0; i < a.size(); i++) {
            newVec[i] = a[i] / b[i];
        }
        return std::move(newVec);
    }
	// get element
	value_type operator[](index_t i) const {
		return vec[i];
	}
	value_type &operator[](index_t i) {
		return vec[i];
	}
	//api
	index_t size() const {
		return vec.size();
	}
    index_t size() {
		return vec.size();
    }
	void display() const {
		cout << "size: ---------" <<size() << "----------" << endl;
        for(index_t i=0; i<size(); i++) {
            cout << vec[i] << " ";
        }
        cout << endl;
	}
	// iterators
	iterator begin() {
		return vec.begin();
	}
	iterator end() {
		return vec.end();
	}
    const_iterator begin() const {
        return vec.begin();
    }
    const_iterator end() const {
        return vec.end();
    }
    friend std::ostream & operator<< (std::ostream &os, const Vec &other) {
        os << "Vec:\t";
        for (uint32_t i = 0; i < other.size(); ++i) {
            os << other[i] << " ";
        }
	return os;
    }
	void randInit(float offset=0.5) {
		float r;
		for(index_t i=0; i<size(); i++) {
			r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - offset);
			vec[i] = r;
		}
	}
    double base() {
        value_type res = 0.0;
        for(vector<value_type>::iterator it=vec.begin(); it!=vec.end(); it++) {
            res += pow(*it, 2);
        }
        return sqrt(res);
    }

private:
	vector<value_type> vec;
};


Vec sqrt(const Vec &vec) {
    Vec tmp(vec);
    for(index_t i = 0; i < vec.size(); i++) {
        tmp[i] = std::sqrt(tmp[i]);
    }
    return std::move(tmp);
}

double dot(const Vec &a, const Vec &b) {
    double res{0.0};
    for(index_t i = 0; i < a.size(); ++i) {
        res += a[i] * b[i];
    }
    return res;
}

}; // end namespace fms

#endif
