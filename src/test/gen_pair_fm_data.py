import sys
import random as r

def gen_fm_features(length, start_id, end_id):
    assert(start_id * end_id > 0)
    ids = set()
    while len(ids) < length:
        id = r.randint(start_id, end_id)
        ids.add(id)
    ids = sorted(ids)
    features = []
    for id in ids:
        fea = r.random()
        features.append(
            "%d:%f" % (id, fea))
    return features

def gen_fm(length, start_id, end_id):
    target = r.random()
    features = gen_fm_features(length, start_id, end_id)
    fm = "%f %s" % (
            target,
            ' '.join(features))
    return fm

def gen_fm_pair():
    common_prefix = " ".join( gen_fm_features(r.randint(2, 8), 1, 100))
    f_fm = gen_fm( r.randint(2, 8), 101, 1000)
    s_fm = gen_fm( r.randint(2, 8), 101, 1000)
    return '\t'.join([common_prefix, f_fm, s_fm])

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print 'USAGE: ./cmd length outpath'
        sys.exit(-1)
    length = int(args[0])
    outpath = args[1]
    
    with open(outpath, 'w') as f:
        for i in xrange(length):
            fm_pair = gen_fm_pair()
            f.write(fm_pair + '\n')
