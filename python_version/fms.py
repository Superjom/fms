# -*- coding: UTF-8 -*-
from __future__ import division
import logging
import sys
import os
from numpy import exp, dot, zeros, outer, random, dtype, get_include, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum,\
    prod, mean, sin, cos, zeros, e, log

class Instance(object):
    '''
    单个训练单位
    '''
    def __init__(self):
        self.target = 0.0
        self.predict = 0.0
        self.features = []

    def add(self, key, value):
        self.features.append( 
            self._make_item(key, value))

    def _make_item(self, key, value):
        return (key, value,)

    def __iter__(self):
        for item in self.features:
            yield item


class Data(object):
    '''
    数据集
    '''
    def __init__(self, path):
        self.path = path
        self.max_val = -sys.maxint
        self.min_val = sys.maxint
        self.max_feature = -sys.maxint
        self.instances = []
        self._init()

    def __getitem__(self, idx):
        return self.instances[idx]

    def size(self):
        return len(self.instances)
    
    def _init(self):
        logging.warning("loading data from\t%s" % self.path)
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                assert(line)
                ins = self._parse_instance(line)
                self.instances.append(ins)

    def _parse_instance(self, line):
        cols = line.strip().split()
        ins = Instance()
        ins.target = float(cols[0])
        for col in cols[1:]:
            key, value = col.split(':')
            key = int(key); value = float(value)
            ins.add(key, value)
            # set status
            self.max_val = max(self.max_val, value)
            self.min_val = min(self.min_val, value)
            self.max_feature = max(self.max_feature, key)
        return ins

class IDData(Data):
    def __init__(self, path):
        Data.__init__(self, path)

    def _parse_instance(self, line):
        cols = line.strip().split()
        ins = Instance()
        ins.target = float(cols[0])
        for col in cols[1:]:
            key = int(col)
            ins.add(key, 1.0)
            self.max_feature = max(self.max_feature, key)
        return ins
         



class FMParam(object):
    '''
    模型参数
    '''
    def __init__(self, dim, num_feas, learning_rate):
        self.dim = dim
        self.num_feas = num_feas
        self.learning_rate = learning_rate
        self.lr_w = empty(self.num_feas, dtype=REAL)
        self.fm_w = empty( (self.num_feas, self.dim), dtype=REAL)
        self.reset_weights()

    def reset_weights(self):
        random.seed(2)
        for i in xrange(self.num_feas):
            self.fm_w[i] = (random.rand(self.dim) - 0.5) / self.dim
        self.lr_w = (random.rand(self.num_feas) - 0.5) / self.num_feas

    def __getitem__(self, idx):
        return (self.lr_w[idx], self.fm_w[idx],)


class AdaGradFMParam(FMParam):
    '''
    利用AdaGrad更新参数
    '''
    def __init__(self, dim, num_feas, learning_rate):
        FMParam.__init__(self, dim, num_feas, learning_rate)
        # for AdaGrad
        self.lr_g2sum = zeros(num_feas)
        self.fm_g2sum = zeros( (self.num_feas, self.dim) )
        # accumulate grad
        self.lr_g = zeros(num_feas)
        self.fm_g = zeros( (self.num_feas, self.dim) )
        # count commits
        self.n = zeros(self.num_feas)

    def batch_commit(self, key, grad):
        '''
        grad = (lr_g, fm_g)
        '''
        #print 'batch_commit>grad', grad
        lr_g, fm_g = grad
        self.lr_g2sum[key] += lr_g ** 2
        self.fm_g2sum[key] += fm_g ** 2
        self.lr_g[key] += lr_g
        self.fm_g[key] += fm_g
        self.n[key] += 1.0

    def batch_push(self):
        '''
        batch update FM parameter
        '''
        def update(param, key, grad, g2sum):
            #print 'g2sum', g2sum
            #print 'grad', grad
            param[key] -= self.learning_rate / sqrt(g2sum) * grad 

        for key in xrange(self.num_feas):
            if self.n[key] > 0.5: 
                update(self.lr_w, key, self.lr_g[key] / self.n[key], self.lr_g2sum[key])
                update(self.fm_w, key, self.fm_g[key] / self.n[key], self.fm_g2sum[key])
        # reset status
        # accumulate grad
        self.lr_g = zeros(self.num_feas)
        self.fm_g = zeros( (self.num_feas, self.dim) )
        # count commits
        self.n = zeros(self.num_feas)


class SGD(object):
    def __init__(self, FMparam):
        self.fm = FMparam

    def learn_instance(self, ins):
        raise NotImplemented


class AdaRMSE_SGD(SGD):
    '''
    RMSE 作为损失函数
    '''
    def __init__(self, FMparam):
        SGD.__init__(self, FMparam)

    def learn_instance(self, ins):
        x_v_sum = zeros(self.fm.dim)
        cost = self.feed_forward(ins, x_v_sum)
        self.feed_backward(ins, x_v_sum)
        return cost

    def feed_forward(self, ins, x_v_sum):
        output = 0.0
        # linear regression
        for key,value in ins:
            output += self.fm.lr_w[key] * value

        x_v_2sum = 0.0
        for key, value in ins:
            x_v = value * self.fm.fm_w[key]
            #print 'x_v', x_v
            x_v_sum += x_v
            x_v_2sum += dot(x_v, x_v)
        output += 0.5 * (dot(x_v_sum, x_v_sum) - x_v_2sum)
        ins.fm_output = output
        ins.predict = 1. / (1. + exp(-output))
        return (ins.predict - ins.target) ** 2

    def feed_backward(self, ins, x_v_sum):
        p, q = ins.target, ins.predict
        grad_RMSE_q = 2 * (q - p)
        y = ins.fm_output
        grad_q_y = exp(y) / ( 1 + exp(y))**2
        for key, value in ins:
            grad_y_wi = value
            grad_y_vi = value * (x_v_sum - value * self.fm.fm_w[key])
            grad_RMSE_wi = grad_RMSE_q * grad_q_y * grad_y_wi
            grad_RMSE_vi = grad_RMSE_q * grad_q_y * grad_y_vi
            grad = (grad_RMSE_wi, grad_RMSE_vi,)
            self.fm.batch_commit(key, grad)


class AdaKLSGD(SGD):
    '''
    利用KL散度作为损失函数
    '''
    def __init__(self, FMparam):
        SGD.__init__(self, FMparam)

    def learn_instance(self, ins):
        x_v_sum = zeros(self.fm.dim)
        cost = self.feed_forward(ins, x_v_sum)
        self.feed_backward(ins, x_v_sum)
        return cost

    def feed_forward(self, ins, x_v_sum):
        output = 0.0
        # linear regression
        for key,value in ins:
            output += self.fm.lr_w[key] * value

        x_v_2sum = 0.0
        for key, value in ins:
            x_v = value * self.fm.fm_w[key]
            #print 'x_v', x_v
            x_v_sum += x_v
            x_v_2sum += dot(x_v, x_v)
        output += 0.5 * (dot(x_v_sum, x_v_sum) - x_v_2sum)
        ins.fm_output = output
        ins.predict = 1. / (1. + exp(-output))
        return  ins.predict * log( ins.predict / ins.target) + \
                ins.target * log( ins.target / ins.predict)

    def feed_backward(self, ins, x_v_sum):
        q = ins.predict
        p = ins.target
        grad_KL_q = - p / q + log(q / p) + 1
        y = ins.fm_output
        grad_q_y = exp(y) / ( 1 + exp(y))**2
        #print 'c', c
        for key, value in ins:
            grad_y_wi = value
            grad_y_vi = value * (x_v_sum - value * self.fm.fm_w[key])
            grad_KL_wi = grad_KL_q * grad_q_y * grad_y_wi
            grad_KL_vi = grad_KL_q * grad_q_y * grad_y_vi
            grad = (grad_KL_wi, grad_KL_vi,)
            self.fm.batch_commit(key, grad)


class Cost(object):
    '''
    统计全局cost
    '''
    def __init__(self):
        self.cost = 0.0
        self.n = 0.0

    def add(self, cost):
        self.cost += cost
        self.n += 1

    def norm(self):
        return self.cost / self.n

    def reset(self):
        self.cost = 0.0
        self.n = 0.0
        

class CrossEntropyFM(object):
    def __init__(self, path, dim, learning_rate=0.1, batch_size=3000):
        self.learning_rate = learning_rate
        self.data = IDData(path)
        self.fm = AdaGradFMParam(dim, self.data.max_feature+1, learning_rate)
        self.sgd = AdaRMSE_SGD(self.fm)
        self.batch_size = batch_size

    def train_iter(self):
        global_cost = Cost()
        for i in xrange(self.data.size()):
            ins = self.data[i]
            cost = self.sgd.learn_instance(ins)
            global_cost.add(cost)

            if i > 0 and i % self.batch_size == 0:
                self.fm.batch_push()
        self.fm.batch_push()
        #logging.warning("cost:\t%f" % float(global_cost.norm()))
        print "cost\t", float(global_cost.norm())


if __name__ == '__main__':
    path = "../tools/1.txt"
    cFM = CrossEntropyFM(path, 10)
    for i in range(100):
        cFM.train_iter()
