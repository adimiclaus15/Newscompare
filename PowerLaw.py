import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math

def power_law(k_min, k_max, y, gamma):
    return ((k_max**(-gamma+1) - k_min**(-gamma+1))*y  + k_min**(-gamma+1.0))**(1.0/(-gamma+1.0))


N = 200
nr_digits = int(math.log10(N))
first_digit = N / (10 ** nr_digits)

k_min = [1, 1, 1, 1, 1]
k_max = [10, 169, 88, 39, 36]
gamma = 1.2
T = 5

f = open("date10.in", "w")
f.write(str(N) + "\n")
f.write(str(T) + "\n")
weight_matrices = []
cost_vertices = []
sum = {}
for t in range(T):
    center = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        y[i] = np.random.uniform(0, 1)
        center[i] = int(power_law(k_min[t], k_max[t], y[i], gamma))
        center[i] -= 1
    centerSort = sorted(center, reverse = True)
    #print(centerSort)
    #plt.plot(range(N), centerSort, 'ro')
    #plt.show()

    influence = []
    for i in range(N):
        influence.append([])
        for j in range(int(center[i])):
            y = np.zeros(N)
            value = np.zeros(N)
            for k in range(N):
                if k == i:
                    value[k] = 0.0
                else:
                    y[k] = np.random.uniform(0, 1)
                    value[k] = power_law(1, 2, y[k], 11)
                    #value[k] = power_law(1, 2, y[k], 11)
                    value[k] -= 1
            influence[i].append(list(value))

    m = np.zeros((N, N))
    cnt = np.zeros((N, N))
    #mx = 0
    for i in range(N):
        for j in range(len(influence[i])):
            #cnt = 0
            for k in range(N):
                if influence[i][j][k] >= 0.5:
                    m[i][k] += influence[i][j][k]
                    cnt[i][k] += 1

    for i in range(N):
        for j in range(N):
            if cnt[i][j] != 0:
                m[i][j] /= cnt[i][j]

    edges = []
    mt = np.transpose(m)
    
    for i in range(N):
        for j in range(i + 1, N):
            if mt[i][j] >= 0.5:
                edges.append((i, j, mt[i][j]))
            if mt[j][i] >= 0.5:
                edges.append((j, i, mt[j][i]))

    print(len(edges))
    D = nx.DiGraph()
    D.add_weighted_edges_from(edges)
    c = nx.pagerank(D)
    for i in range(N):
        if i not in c:
            c[i] = 0
        c[i] *= (first_digit * (10 ** (nr_digits + 1)))
    for x in c:
        if x not in sum:
            sum[x] = 0
        sum[x] += c[x]
    cost_vertices.append(c)
    weight_matrices.append(m)

for i in range(N):
    f.write(str(sum[i] / T) + " ")
f.write("\n")


for t in range(T):
    f.write(str(t) + "\n")
    for i in range(N):
        for j in range(i + 1, N):
            if weight_matrices[t][i][j] >= 0.5:
                f.write(str(i) + " " + str(j) + " " + str(weight_matrices[t][i][j]) + "\n")
            if weight_matrices[t][j][i] >= 0.5:
                f.write(str(j) + " " + str(i) + " " + str(weight_matrices[t][j][i]) + "\n")