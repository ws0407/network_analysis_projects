import numpy as np

ROOT_DIR = '/tmp/EBSNN/'
one_sample = [
    np.load(ROOT_DIR + 'data/sample.npy'),
    np.load(ROOT_DIR + 'data/label.npy')
]
x = one_sample[0]
y = one_sample[1]

print(x)
print(y)

