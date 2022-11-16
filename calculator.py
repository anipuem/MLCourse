# import numpy as np
# kernel = np.array([0.267,-0.167,0,-0.167,0.367,-0.2,0,-0.2,0.6]).reshape((3, 3))
# print(kernel)
# print(np.linalg.inv(kernel))

# import numpy as np
# from sympy import *
#
# a1 = Symbol('a1')
# a2 = Symbol('a2')
# a3 = Symbol('a3')
# a5 = Symbol('a5')
# a6 = Symbol('a6')
# m = np.array([[-2,-2,1,1], [1,1,-1,-1], [1,2,-1,0], [1,0,0,-1]])
# num = np.dot(m, np.array([a1, a3, a5, a6]))
#
# f1 = num[0]+4
# f2 = num[1]
# f3 = num[2]-2
# f5 = num[3]-2
# r = solve([f1,f2,f3,f5], [a1,a3,a5,a6])
# print(r)
# # n = np.array([2, 2, 2, 0, 0])   # 可替换为式子右边的常数
# # solution = np.linalg.solve(m, n)   #solution format: np.array([x, y, z])

# sizes = [3,4,2]
# for ch1, ch2 in zip(sizes[:-1], sizes[1:]):
#     print('aaaaa')
#     print(ch1,ch2)
#
# for ch in sizes[1:]:
#     print(ch)

# list1 = [[3,3,3,3], [4,4]]
# seq3 = [[[1,1,1,1],[1,1,1,1],[1,1,1,1]],[[2,2],[2,2]]]
# for i, (a, b) in enumerate(zip(list1, seq3)):
#     print('{}：({},{})'.format(i, a, b))
#     print(i)
# num_layers = 4
# for l in range(num_layers - 2, -1, -1):
#     print(l)
#
# print(21%10)

# from scipy import stats
# data = stats.gamma.rvs(2, loc=1.5, scale=2, size=10000)
# from fitter import Fitter
# f = Fitter(data, distributions=['lognorm', 'expon', 'exponpow', 'norm'])
# f.fit()
# # may take some time since by default, all distributions are tried
# # but you call manually provide a smaller set of distributions
# print(f.summary())

# import os
# if os.path.exists('.\disfiga'):
#     print('aa')
# else:
#     os.mkdir('.\disfiga')



