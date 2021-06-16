from random import random, randint
import numpy as np
import matplotlib.pyplot as plt
from time import time

class Circuit:
    def __init__(self, id, current, work_time=0):
        self.id = id
        self.current = current
        self.work_time = work_time

class TimeSlice:
    def __init__(self, id, current=0):
        self.id = id
        self.current = current

class TempController:
    def __init__(self, circuitNum, timesliceNum):
        self.circuitNum, self.timesliceNum = circuitNum, timesliceNum
        self.circuits = [Circuit(i, randint(10, 100)/100) for i in range(circuitNum)]
        self.circuits.sort(key=lambda x: -x.current)
        self.timeSlices = []
        # 路径最大边
        self.edge = None

    def setup(self):
        # 获取上个时间片末尾      
        if self.timeSlices:
            self.obj = self.timeSlices[-1]
            for cir in self.circuits:
                cir.work_time = cir.work_time + round(np.random.randn(1)[0]) 
                cir.work_time = min(max(cir.work_time, 0), self.timesliceNum)
        else:
            self.obj = TimeSlice(-1, 0)
            for cir in self.circuits:
                cir.work_time = randint(0, self.timesliceNum)  
        # 初始化时间片
        self.timeSlices = [TimeSlice(i) for i in range(self.timesliceNum)]
        self.timeslice_occupy = np.zeros((self.timesliceNum, self.circuitNum), dtype=np.bool)

    def greedy(self):
        for cir in self.circuits:
            for i in range(cir.work_time):
                self.timeSlices[i].current += cir.current
                self.timeslice_occupy[self.timeSlices[i].id][cir.id] = True
            if (cir.work_time != 0) and (cir.work_time != self.timesliceNum):
                self.timeSlices = merge_sort(self.timeSlices, cir.work_time - 1, key=lambda x: x.current)

    def no_algorithm(self):
        ts_na = [TimeSlice(i) for i in range(self.timesliceNum)]
        for cir in self.circuits:
            for i in range(cir.work_time):
                ts_na[i].current += cir.current
        return [ts.current for ts in ts_na]

    def groups(self):
        ts_na = [TimeSlice(i) for i in range(self.timesliceNum)]
        start = 0
        for cir in self.circuits:
            for i in range(cir.work_time):
                ts_na[(start+i)%self.timesliceNum].current += cir.current
            start += 1
        return [ts.current for ts in ts_na]

    def loop(self):
        self.setup()
        # problem 1
        self.greedy()
        # problem 2
        self.timeSlices, q = insert_sort(self.timeSlices, self.obj, key=lambda x: x.current)
        self.timeSlices, self.edge = dyn_prog(q, self.timeSlices, key=lambda x: x.current)
        return [ts.current for ts in self.timeSlices][1:]
    
    def test(self):
        if self.edge is not None:
            print("diff:", max([abs(self.timeSlices[i+1].current-self.timeSlices[i].current) for i in range(len(self.timeSlices)-1)])-self.edge)


# 每个循环运行一遍
def merge_sort(objs, mid, key=lambda x:x):
    new_objs = []
    istart, iend, jstart, jend = 0, mid, mid+1, len(objs)-1
    while (istart <= iend) and (jstart <= jend):
        if key(objs[istart]) < key(objs[jstart]):
            new_objs.append(objs[istart])
            istart += 1
        else:
            new_objs.append(objs[jstart])
            jstart += 1
    if istart <= iend:
        new_objs.extend(objs[istart:iend+1])
    else:
        new_objs.extend(objs[jstart:jend+1])
    return new_objs

def insert_sort(objs, insert_obj, key):
    insert_value = key(insert_obj)
    for index, obj in enumerate(objs):
        if insert_value < key(obj):
            break
    objs.insert(index, obj)
    return objs, index

def dyn_prog(q, objs, key):
    if q==0:
        return objs, None
    if (q == len(objs)-1):
        objs.reverse()
        return objs, None
    edge_valley, path_valley = calc_path(q, objs, key)
    objs.reverse()
    edge_peak, path_peak = calc_path(len(objs)-1-q, objs, key)
    new_objs = []
    if(edge_peak <= edge_valley):
        for i in range(len(objs)):
            new_objs.append(objs[path_peak[i]])
        return new_objs, edge_peak
    else:
        objs.reverse()
        for i in range(len(objs)):
            new_objs.append(objs[path_valley[i]])
        return new_objs, edge_valley


def calc_path(q, objs, key):
    max_edge = np.zeros((len(objs), len(objs)), dtype=np.float)
    max_edge[0][1] = abs(key(objs[1]) - key(objs[0]))
    path = np.zeros((len(objs), len(objs), len(objs)), dtype=np.int)
    path[0][1][0:2] = (0, 1)
    for i in range(q+1):
        for j in range(len(objs) if i==q else q+1):
            lmax = max(i, j)
            if i>j:
                max_edge[i][j] = max_edge[j][i]
                for l in range(lmax+1):
                    path[i][j][l] = path[j][i][lmax - l]
            elif i<j-1:
                max_edge[i][j] = max(max_edge[i][j - 1], abs(key(objs[j]) - key(objs[j - 1])))
                for l in range(lmax):
                    path[i][j][l] = path[i][j - 1][l]
                path[i][j][lmax] = j               
            elif i == (j - 1) and i != 0:
                kmin = 0
                kmin_edge = max(max_edge[i][kmin], abs(key(objs[j]) - key(objs[kmin])))
                for k in range(1, i):
                    k_edge = max(max_edge[i][k], abs(key(objs[j]) - key(objs[k])))
                    if (k_edge < kmin_edge):
                        kmin_edge = k_edge
                        kmin = k
                max_edge[i][j] = kmin_edge
                for l in range(lmax):
                    path[i][j][l] = path[i][kmin][l]
                path[i][j][lmax] = j
    return max_edge[q][len(objs)-1], path[q][len(objs)-1][:]


def one_test(circuitNum, timesliceNum):
    tc = TempController(circuitNum, timesliceNum)
    curs_na = []
    curs_group = []
    curs_greedy = []
    for i in range(10):
        curs_greedy.extend(tc.loop())
        curs_na.extend(tc.no_algorithm())
        curs_group.extend(tc.groups())
    
    def draw_and_save():
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.plot(list(range(len(curs_na))), curs_na,'k:', label='initial design')
        ax.plot(list(range(len(curs_group))), curs_group,'k--', label='grouping design')
        ax.plot(list(range(len(curs_greedy))), curs_greedy,'k-', label='this paper')
        plt.xlabel('time slice')
        plt.ylabel('current')

        ax.legend(loc="lower right")
        plt.savefig("ts_current.png")

    def sat():
        max_change_init = max([abs(curs_na[i]-curs_na[i+1]) for i in range(len(curs_na)-1)])
        range_init = max(curs_na)-min(curs_na)
        max_change_group = max([abs(curs_group[i]-curs_group[i+1]) for i in range(len(curs_group)-1)])
        range_group = max(curs_group)-min(curs_group)
        max_change_greedy = max([abs(curs_greedy[i]-curs_greedy[i+1]) for i in range(len(curs_greedy)-1)])
        range_greedy = max(curs_greedy)-min(curs_greedy) 
        return max_change_greedy/max_change_init, max_change_greedy/max_change_group, range_greedy/range_init, range_greedy/range_group

    return sat()

def main(circuitNum, timesliceNum, test_time=100):
    start = time()
    change_per_inits, change_per_groups, range_per_inits, range_per_groups = 0, 0, 0, 0
    for i in range(test_time):
        change_per_init, change_per_group, range_per_init, range_per_group = one_test(circuitNum, timesliceNum)
        change_per_inits += change_per_init
        change_per_groups += change_per_group
        range_per_inits += range_per_init
        range_per_groups += range_per_group
    end = time()
    print("===sat result===")
    print(change_per_inits/test_time, change_per_groups/test_time, range_per_inits/test_time, range_per_groups/test_time)
    print((end-start)/test_time/10)

if __name__ == "__main__":
    main(100, 10)
