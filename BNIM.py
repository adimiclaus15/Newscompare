from typing import NamedTuple
from gurobipy import *
import numpy as np
from datetime import datetime

class TemporalEdge(NamedTuple):
    x: int
    y: int
    cost: float
    time: int

def read_temporal_graph(file_name, directed = False):
    adj_list = []
    with open(file_name) as f:
        n = int(f.readline()) #number of vertices
        T = int(f.readline()) #number of snapshots (lifespan)
        #cost for each vertex in xeach snapshot
        #cost_vertices = [[float(x) for x in f.readline().split()] for i in range(T)]
        cost_vertices = [float(x) for x in f.readline().split()]
        #weight_matrix=[[[float("inf") for i in range(n)] for j in range(n)] for t in range(T)]
        weight_matrix = [[[0 for i in range(n)] for j in range(n)] for t in range(T)]
        for line in f:
            #read snapshot t, with t = 0,...,T - 1
            list_edge = line.split()
            if len(list_edge) > 1: #is not timpstamp - it is an edge with cost
                e = TemporalEdge(x = int(list_edge[0]), y = int(list_edge[1]), cost = float(list_edge[2]), time = t)
                adj_list.append(e)
                weight_matrix[e.time][e.x][e.y] = e.cost
                if not directed:
                    weight_matrix[e.time][e.y][e.x] = e.cost
            else:
                t = int(list_edge[0])

    cost_vertices = np.array(cost_vertices)
    weight_matrix = np.array(weight_matrix)
    return n, T, adj_list, cost_vertices, weight_matrix

def getFiles(directory_path):
    file_paths = []
    for root, directories, files in os.walk(directory_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths

def getMinMeanMax(yt):
    y_mean = [0 for i in range(3)]
    y_max = [0 for i in range(3)]
    y_min = [1000000 for i in range(3)]
    for i in range(3):
        for j in range(len(yt)):
            y_mean[i] += yt[j][i]
            y_max[i] = max(y_max[i], yt[j][i])
            y_min[i] = min(y_min[i], yt[j][i])
        y_mean[i] /= len(yt)
    print(y_min)
    print(y_mean)
    print(y_max)

def L11a():
    files = getFiles('/home/adi/Desktop/newscompare/200/')
    v = []
    p = []
    idx = 0
    for file in files:
        idx += 1
        print(idx)
        n, T, adj_list, cost_vertices, W = read_temporal_graph(file, True)
        total_influence = np.sum(W)
        S = sum(cost_vertices)
        influence = []
        timp = []
        for step in range(4, 7): #change budget
            #print(step)
            start = datetime.now()
            P = step / 10
            B = P * S
            model = Model()
            model.Params.LogToConsole = 0
            x = model.addMVar(n, vtype = GRB.BINARY, name = "x")
            model.addConstr(x @ cost_vertices <= B)
            obj = []
            for j in range(n):
                for t in range(T):
                    for i in range(n):
                        obj.append(x[i] * W[t, i, j])
            model.setObjective(sum(obj), GRB.MAXIMIZE)
            model.update()
            model.optimize()
            end = datetime.now()
            influence.append(round((model.objVal / total_influence) * 100, 2))
            timp.append((end - start).total_seconds())
            #print((end - start).total_seconds())
            #model.write("model.lp")
            # for i in range(n):
            #     print(x[i].x)
        v.append(influence)
        p.append(timp)
        #print(influence)
    getMinMeanMax(p)
    

def L11b():
    files = getFiles('/home/adi/Desktop/newscompare/200/')
    #print(files)
    v = []
    p = []
    idx = 0
    for file in files:
        idx += 1
        print(idx)
        n, T, adj_list, cost_vertices, W = read_temporal_graph(file, True)
        total_influence = np.sum(W)
        S = sum(cost_vertices)
        sol = []
        timp = []
        for step in range(4, 7):
            start = datetime.now()
            P = step / 10
            B = P * S
            model = Model()
            model.Params.LogToConsole = 0
            x = model.addMVar(n, vtype = GRB.BINARY, name = "x")
            model.addConstr(x @ cost_vertices <= B)
            obj = []
            for j in range(n):
                for t in range(T):
                    for i in range(n):
                        obj.append(x[i] * W[t, i, j] * (1 - x[j]))
            model.setObjective(sum(obj), GRB.MAXIMIZE)
            model.update()
            model.optimize()
            end = datetime.now()
            sol.append(round(model.objVal, 2))
            timp.append((end - start).total_seconds())
            #influence.append(round((model.objVal / total_influence) * 100, 2))
        v.append(sol)
        p.append(timp)
        for i in range(len(sol)):
            sol[i] = (sol[i] / sol[-1]) * 100
        #print(sol)
    getMinMeanMax(p)

def LINFINFa():
    files = getFiles('/home/adi/Desktop/newscompare/200/')
    v = []
    p = []
    idx = 0
    for file in files:
        idx += 1
        print(idx)
        n, T, adj_list, cost_vertices, W = read_temporal_graph(file, True)
        S = sum(cost_vertices)
        sol = []
        timp = []
        for step in range(1, 11):
            start = datetime.now()
            P = step / 10
            B = P * S
            model = Model()
            model.Params.LogToConsole = 0
            x = model.addMVar(n, vtype = GRB.BINARY, name = "x")
            model.addConstr(x @ cost_vertices <= B)
            obj = {}
            for j in range(n):
                for t in range(T):
                    influence = []
                    for i in range(n):
                        influence.append(x[i] * W[t, i, j])
                    obj[(j, t)] = model.addVar()
                    model.addConstr(obj[(j, t)] == sum(influence))
            y = model.addVar()
            model.addConstr(y == max_(obj.values()))
            model.setObjective(y, GRB.MAXIMIZE)
            model.update()
            model.optimize()
            end = datetime.now()
            sol.append(round(y.X, 2))
            timp.append((end - start).total_seconds())
        for i in range(len(sol)):
            sol[i] = (sol[i] / sol[-1]) * 100
        #print(sol)
        p.append(timp)
    getMinMeanMax(p)
            

def LINFINFb():
    files = getFiles('/home/adi/Desktop/newscompare/200/')
    #v = []
    p = []
    idx = 0
    for file in files:
        idx += 1
        print(idx)
        n, T, adj_list, cost_vertices, W = read_temporal_graph(file, True)
        S = sum(cost_vertices)
        sol = []
        timp = []
        for step in range(4, 7):
            start = datetime.now()
            P = step / 10
            B = P * S
            model = Model()
            model.Params.LogToConsole = 0
            x = model.addMVar(n, vtype = GRB.BINARY, name = "x")
            model.addConstr(x @ cost_vertices <= B)
            obj = {}
            for j in range(n):
                for t in range(T):
                    influence = []
                    for i in range(n):
                        influence.append(x[i] * W[t, i, j] * (1 - x[j]))
                    obj[(j, t)] = model.addVar()
                    model.addConstr(obj[(j, t)] == sum(influence))
            y = model.addVar()
            model.addConstr(y == max_(obj.values()))
            model.setObjective(y, GRB.MAXIMIZE)
            model.update()
            model.optimize()
            end = datetime.now()
            sol.append(round(y.X, 2))
            timp.append((end - start).total_seconds())
        for i in range(len(sol)):
            sol[i] = (sol[i] / sol[-1]) * 100
        p.append(timp)
    getMinMeanMax(p)

def LINFa():
    p = []
    idx = 0
    files = getFiles('/home/adi/Desktop/newscompare/200/')
    for file in files:
        idx += 1
        print(idx)
        n, T, adj_list, cost_vertices, W = read_temporal_graph(file, True)
        S = sum(cost_vertices)
        sol = []
        timp = []
        for step in range(4, 7):
            start = datetime.now()
            P = step / 10
            B = P * S
            #B = 2 #set budget
            model = Model()
            model.Params.LogToConsole = 0
            x = model.addMVar(n, vtype = GRB.BINARY, name = "x")
            model.addConstr(x @ cost_vertices <= B)
            obj = {}
            line = []
            for j in range(n):
                timestamp = []
                for t in range(T):
                    influence = []
                    for i in range(n):
                        influence.append(x[i] * W[t, i, j])
                    obj[(j, t)] = model.addVar()
                    model.addConstr(obj[(j, t)] == sum(influence))
                    timestamp.append(obj[(j, t)])
                line.append(sum(timestamp))
            #print(line)
            z = {}
            for i in range(len(line)):
                z[i] = model.addVar()
                model.addConstr(z[i] == line[i])
            y = model.addVar()
            model.addConstr(y == max_(z.values()))
            model.setObjective(y, GRB.MAXIMIZE)
            model.update()
            model.optimize()
            end = datetime.now()
            sol.append(round(y.X, 2))
            timp.append((end - start).total_seconds())
        for i in range(len(sol)):
            sol[i] = (sol[i] / sol[-1]) * 100
        #print(sol)
        p.append(timp)
    getMinMeanMax(p)

def LINFb():
    p = []
    idx = 0
    files = getFiles('/home/adi/Desktop/newscompare/200/')
    for file in files:
        idx += 1
        print(idx)
        n, T, adj_list, cost_vertices, W = read_temporal_graph(file, True)
        S = sum(cost_vertices)
        sol = []
        timp = []
        for step in range(4, 7):
            start = datetime.now()
            P = step / 10
            B = P * S
            #B = 2 #set budget
            model = Model()
            model.Params.LogToConsole = 0
            x = model.addMVar(n, vtype = GRB.BINARY, name = "x")
            model.addConstr(x @ cost_vertices <= B)
            obj = {}
            line = []
            for j in range(n):
                timestamp = []
                for t in range(T):
                    influence = []
                    for i in range(n):
                        influence.append(x[i] * W[t, i, j] * (1 - x[j]))
                    obj[(j, t)] = model.addVar()
                    model.addConstr(obj[(j, t)] == sum(influence))
                    timestamp.append(obj[(j, t)])
                line.append(sum(timestamp))
            z = {}
            for i in range(len(line)):
                z[i] = model.addVar()
                model.addConstr(z[i] == line[i])
            y = model.addVar()
            model.addConstr(y == max_(z.values()))
            model.setObjective(y, GRB.MAXIMIZE)
            model.update()
            model.optimize()
            end = datetime.now()
            sol.append(round(y.X, 2))
            timp.append((end - start).total_seconds())
        for i in range(len(sol)):
            sol[i] = (sol[i] / sol[-1]) * 100
        #print(sol)
        p.append(timp)
    getMinMeanMax(p)

def L1INFa():
    files = getFiles('/home/adi/Desktop/newscompare/200/')
    v = []
    p = []
    idx = 0
    for file in files:
        idx += 1
        print(idx)
        n, T, adj_list, cost_vertices, W = read_temporal_graph(file, True)
        S = sum(cost_vertices)
        sol = []
        timp = []
        for step in range(4, 7):
            start = datetime.now()
            P = step / 10
            B = P * S
            #B = 2 #set budget
            model = Model()
            model.Params.LogToConsole = 0
            x = model.addMVar(n, vtype = GRB.BINARY, name = "x")
            model.addConstr(x @ cost_vertices <= B)
            obj = {}
            line = []
            for j in range(n):
                timestamp = []
                for t in range(T):
                    influence = []
                    for i in range(n):
                        influence.append(x[i] * W[t, i, j])
                    obj[(j, t)] = model.addVar()
                    model.addConstr(obj[(j, t)] == sum(influence))
                    timestamp.append(obj[(j, t)])
                y = model.addVar()
                model.addConstr(y == max_(timestamp))
                line.append(y)
            #print(line)
            model.setObjective(sum(line), GRB.MAXIMIZE)
            model.update()
            model.optimize()
            end = datetime.now()
            sol.append(round(model.objVal, 2))
            timp.append((end - start).total_seconds())
        for i in range(len(sol)):
            sol[i] = (sol[i] / sol[-1]) * 100
        #print(sol)
        p.append(timp)
    getMinMeanMax(p)

def L1INFb():
    files = getFiles('/home/adi/Desktop/newscompare/50_1/')
    p = []
    idx = 0
    for file in files:
        idx += 1
        print(idx)
        n, T, adj_list, cost_vertices, W = read_temporal_graph(file, True)
        S = sum(cost_vertices)
        sol = []
        timp = []
        for step in range(4, 7):
            start = datetime.now()
            P = step / 10
            B = P * S
            model = Model()
            model.Params.LogToConsole = 0
            x = model.addMVar(n, vtype = GRB.BINARY, name = "x")
            model.addConstr(x @ cost_vertices <= B)
            obj = {}
            line = []
            for j in range(n):
                timestamp = []
                for t in range(T):
                    influence = []
                    for i in range(n):
                        influence.append(x[i] * W[t, i, j] * (1 - x[j]))
                    obj[(j, t)] = model.addVar()
                    model.addConstr(obj[(j, t)] == sum(influence))
                    timestamp.append(obj[(j, t)])
                y = model.addVar()
                model.addConstr(y == max_(timestamp))
                line.append(y)
            model.setObjective(sum(line), GRB.MAXIMIZE)
            model.update()
            model.optimize()
            end = datetime.now()
            sol.append(round(model.objVal, 2))
            timp.append((end - start).total_seconds())
        for i in range(len(sol)):
            sol[i] = (sol[i] / sol[-1]) * 100
        #print(sol)
        p.append(timp)
    getMinMeanMax(p)

if __name__ == "__main__":
    L1INFb()
