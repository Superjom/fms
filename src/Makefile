CC=gcc
#CXX="/home/work/.jumbo/opt/gcc48/bin/g++"
CXX=g++
BIN="../bin"
CXXFLAGS= -std=c++11 \
    -pthread \
    -Wall \
    -g \
    -O3


THIRD_INCPATH=-I../third/local/glog/include

THIRD_LIB=../third/local/glog/lib/libglog.a


.PHONY:all test clean

all: kl_fm_train.out kl_fm_predict.out pair_fm_train.out pair_sigmoid_fm_train.out

kl_fm_train.out: kl_fm_train.cpp core/* utils/* data.h
	mkdir -p $(BIN)
	$(CXX) kl_fm_train.cpp $(CXXFLAGS) $(THIRD_INCPATH) -Xlinker  $(THIRD_LIB)  -o kl_fm_train.out
	mv kl_fm_train.out $(BIN)

kl_fm_predict.out: kl_fm_predict.cpp core/* utils/* data.h
	mkdir -p $(BIN)
	$(CXX) kl_fm_predict.cpp $(CXXFLAGS) $(THIRD_INCPATH) -Xlinker  $(THIRD_LIB)  -o kl_fm_predict.out
	mv kl_fm_predict.out $(BIN)

pair_fm_train.out: pair_fm_train.cpp core/* utils/* data.h
	mkdir -p $(BIN)
	$(CXX) pair_fm_train.cpp $(CXXFLAGS) $(THIRD_INCPATH) -Xlinker  $(THIRD_LIB)  -o pair_fm_train.out
	mv pair_fm_train.out $(BIN)

pair_sigmoid_fm_train.out: pair_sigmoid_fm_train.cpp core/* utils/* data.h
	mkdir -p $(BIN)
	$(CXX) pair_sigmoid_fm_train.cpp $(CXXFLAGS) $(THIRD_INCPATH) -Xlinker  $(THIRD_LIB)  -o pair_sigmoid_fm_train.out
	mv pair_sigmoid_fm_train.out $(BIN)

test: test_data.out test_split.out test_kl.out test_vec.out test_ListData.out

#test_FMValue.out: test/test_FMValue.cpp core/FMCParam.h utils/vec.h
	#$(CXX) test/test_FMValue.cpp $(CXXFLAGS) $(THIRD_INCPATH) -Xlinker  $(THIRD_LIB)  -o test_FMValue.out
	#mv test_FMValue.out $(BIN)

test_data.out: test/test_data.cpp data.h core/common.h utils/all.h utils/common.h utils/string.h 
	$(CXX) test/test_data.cpp $(CXXFLAGS) $(THIRD_INCPATH) -Xlinker  $(THIRD_LIB)  -o test_data.out
	mv test_data.out test/

test_kl.out: test/test_kl.cpp data.h core/common.h utils/all.h utils/common.h utils/string.h 
	$(CXX) test/test_kl.cpp $(CXXFLAGS) $(THIRD_INCPATH) -Xlinker  $(THIRD_LIB)  -o test_kl.out
	mv test_kl.out test/

test_split.out: test/test_split.cpp  core/common.h utils/all.h utils/common.h utils/string.h 
	$(CXX) test/test_split.cpp $(CXXFLAGS) $(THIRD_INCPATH) -Xlinker  $(THIRD_LIB)  -o test_split.out
	mv test_split.out test/

test_FMCParam.out: test/test_FMCParam.cpp core/FMCParam.h utils/all.h utils/common.h utils/string.h 
	$(CXX) test/test_FMCParam.cpp $(CXXFLAGS) $(THIRD_INCPATH) -Xlinker  $(THIRD_LIB)  -o test_FMCParam.out
	mv test_FMCParam.out test/

test_vec.out: test/test_vec.cpp utils/all.h utils/vec.h
	$(CXX) test/test_vec.cpp $(CXXFLAGS) $(THIRD_INCPATH) -Xlinker  $(THIRD_LIB)  -o test_vec.out
	mv test_vec.out test/

test_ListData.out: test/test_ListData.cpp data.h
	$(CXX) test/test_ListData.cpp $(CXXFLAGS) $(THIRD_INCPATH) -Xlinker  $(THIRD_LIB)  -o test_ListData.out
	mv test_ListData.out test/


clean:
	rm *.out
	rm $(BIN)/*


