import bisect
import sys
import time
import math
import numpy as np
import functools
import random
import copy
# import matplotlib
# import matplotlib.pyplot as plt


class demand_node(object):
    def __init__(self, node_a, node_b, demand, cost):
        self.a = node_a
        self.b = node_b
        self.demand = demand
        self.cost = cost

    def __str__(self):
        return "({}, {}) = {}".format(self.a, self.b, self.demand)

    def __lt__(self, other):
        return self.demand < other.demand

    def distance_to_cur(self, cur_pos):
        dist = cost_graph[self.a][cur_pos]
        if cost_graph[self.b][cur_pos] < dist:
            dist = cost_graph[self.b][cur_pos]
        return dist


def print_list(list):
    for e in list:
        print_route(e)
        print(cal_cost(e))
    print()


def print_route(list):
    res = "s "
    for l in list:
        res += "0,"
        for e in l:
            res = res + "(" + str(e.a + 1) + "," + str(e.b + 1) + "),"
        res += "0,"
    res = res[:-1]
    print(res)


def closest_jobs(demand_list, cur_pos, cur_cap, reverse_flag=False, value_flag=False):
    def dist_cmp(e1, e2):
        return e1[1] - e2[1]

    cal_list = []
    dist_list = []
    for job in demand_list:
        if job.demand > cur_cap: continue
        dist = job.distance_to_cur(cur_pos)
        cal_list.append((job, dist))
        if value_flag:
            if dist == 0:
                dist_list.append(np.inf)
            else:
                dist_list.append(job.demand / dist)
        else:
            dist_list.append(dist)
    if len(dist_list) == 0: return []
    cal_list.sort(key=functools.cmp_to_key(dist_cmp), reverse=reverse_flag)
    dist_list.sort(reverse=reverse_flag)
    index = bisect.bisect(dist_list, dist_list[0])
    return cal_list[0:index]


def further_choose_jobs(job_list, depot, reverse_flag=False, value_flag=False):
    dm_list = []
    for job_and_dist in job_list:
        dm_list.append(job_and_dist[0])
    res = closest_jobs(dm_list, depot, np.inf, reverse_flag, value_flag)
    return res


# 1. 随机选一个最近的任务
# 2. 最远
# 3. 最近
# 4. 最大demand/cost
# 5. 最小demand/cost
# 6. 前一半最远，后一半最近
# 在六种里面选择最优返回
def path_scanning(original_job_list):
    local_demand_list = copy.deepcopy(original_job_list)
    # -------------------随机一个最近-------------------
    route_list = []
    while len(local_demand_list) > 0:  # 重开, start at depot
        cur_route = []
        cur_cap = capacity
        cur_pos = depot
        while cur_cap > 0 and len(local_demand_list) > 0:
            job_list = closest_jobs(local_demand_list, cur_pos, cur_cap)
            if len(job_list) == 0: break
            # more than one choice, apply five rules
            original_job = job_list[0][0]
            job = job_list[0][0]
            if cost_graph[cur_pos][job.b] < cost_graph[cur_pos][job.a]:
                job = demand_node(job.b, job.a, job.demand, job.cost)
            cur_pos = job.b
            local_demand_list.remove(original_job)
            cur_cap -= job.demand
            cur_route.append(job)
        route_list.append(cur_route)
    optimal_route = route_list
    optimal_cost = cal_cost(route_list)
    # print(optimal_cost)

    # --------------------最远----------------------
    route_list = []
    local_demand_list = copy.deepcopy(original_job_list)
    while len(local_demand_list) > 0:  # 重开, start at depot
        cur_route = []
        cur_cap = capacity
        cur_pos = depot
        while cur_cap > 0 and len(local_demand_list) > 0:
            job_list = closest_jobs(local_demand_list, cur_pos, cur_cap)
            if len(job_list) == 0: break
            # more than one choice, apply five rules
            if len(job_list) > 1:
                job_list = further_choose_jobs(job_list, depot, True, False)
            original_job = job_list[0][0]
            job = job_list[0][0]
            if cost_graph[cur_pos][job.b] < cost_graph[cur_pos][job.a]:
                job = demand_node(job.b, job.a, job.demand, job.cost)
            cur_pos = job.b
            local_demand_list.remove(original_job)
            cur_cap -= job.demand
            cur_route.append(job)
        route_list.append(cur_route)
    val = cal_cost(route_list)
    # print(val)
    if val < optimal_cost:
        # print("update to 2")
        optimal_cost = val
        optimal_route = route_list

    # --------------------最近----------------------
    route_list = []
    local_demand_list = copy.deepcopy(original_job_list)
    while len(local_demand_list) > 0:  # 重开, start at depot
        cur_route = []
        cur_cap = capacity
        cur_pos = depot
        while cur_cap > 0 and len(local_demand_list) > 0:
            job_list = closest_jobs(local_demand_list, cur_pos, cur_cap)
            if len(job_list) == 0: break
            # more than one choice, apply five rules
            if len(job_list) > 1:
                job_list = further_choose_jobs(job_list, depot, False, False)
            original_job = job_list[0][0]
            job = job_list[0][0]
            if cost_graph[cur_pos][job.b] < cost_graph[cur_pos][job.a]:
                job = demand_node(job.b, job.a, job.demand, job.cost)
            cur_pos = job.b
            local_demand_list.remove(original_job)
            cur_cap -= job.demand
            cur_route.append(job)
        route_list.append(cur_route)
    val = cal_cost(route_list)
    # print(val)
    if val < optimal_cost:
        # print("update to 3")
        optimal_cost = val
        optimal_route = route_list

    # --------------------demand/cost最大----------------------
    route_list = []
    local_demand_list = copy.deepcopy(original_job_list)
    while len(local_demand_list) > 0:  # 重开, start at depot
        cur_route = []
        cur_cap = capacity
        cur_pos = depot
        while cur_cap > 0 and len(local_demand_list) > 0:
            job_list = closest_jobs(local_demand_list, cur_pos, cur_cap)
            if len(job_list) == 0: break
            # more than one choice, apply five rules
            if len(job_list) > 1:
                job_list = further_choose_jobs(job_list, depot, True, True)
            original_job = job_list[0][0]
            job = job_list[0][0]
            if cost_graph[cur_pos][job.b] < cost_graph[cur_pos][job.a]:
                job = demand_node(job.b, job.a, job.demand, job.cost)
            cur_pos = job.b
            local_demand_list.remove(original_job)
            cur_cap -= job.demand
            cur_route.append(job)
        route_list.append(cur_route)
    val = cal_cost(route_list)
    # print(val)
    if val < optimal_cost:
        # print("update to 4")
        optimal_cost = val
        optimal_route = route_list

    # --------------------demand/cost最小----------------------
    route_list = []
    local_demand_list = copy.deepcopy(original_job_list)
    while len(local_demand_list) > 0:  # 重开, start at depot
        cur_route = []
        cur_cap = capacity
        cur_pos = depot
        while cur_cap > 0 and len(local_demand_list) > 0:
            job_list = closest_jobs(local_demand_list, cur_pos, cur_cap)
            if len(job_list) == 0: break
            # more than one choice, apply five rules
            if len(job_list) > 1:
                job_list = further_choose_jobs(job_list, depot, False, True)
            original_job = job_list[0][0]
            job = job_list[0][0]
            if cost_graph[cur_pos][job.b] < cost_graph[cur_pos][job.a]:
                job = demand_node(job.b, job.a, job.demand, job.cost)
            cur_pos = job.b
            local_demand_list.remove(original_job)
            cur_cap -= job.demand
            cur_route.append(job)
        route_list.append(cur_route)
    val = cal_cost(route_list)
    # print(val)
    if val < optimal_cost:
        # print("update to 5")
        optimal_cost = val
        optimal_route = route_list

    # --------------half related------------
    route_list = []
    local_demand_list = copy.deepcopy(original_job_list)
    while len(local_demand_list) > 0:  # 重开, start at depot
        cur_route = []
        cur_cap = capacity
        cur_pos = depot
        while cur_cap > 0 and len(local_demand_list) > 0:
            job_list = closest_jobs(local_demand_list, cur_pos, cur_cap)
            if len(job_list) == 0: break
            # more than one choice, apply five rules
            if len(job_list) > 1:
                if cur_cap > capacity / 2:  # less than full
                    job_list = further_choose_jobs(job_list, depot, True, False)
                else:
                    job_list = further_choose_jobs(job_list, depot, False, False)
            original_job = job_list[0][0]
            job = job_list[0][0]
            if cost_graph[cur_pos][job.b] < cost_graph[cur_pos][job.a]:
                job = demand_node(job.b, job.a, job.demand, job.cost)
            cur_pos = job.b
            local_demand_list.remove(original_job)
            cur_cap -= job.demand
            cur_route.append(job)
        route_list.append(cur_route)
        # print(val)
    if val < optimal_cost:
        # print("update to 6")
        optimal_cost = val
        optimal_route = route_list
    return optimal_route


# 计算全局cost
def cal_cost(route_list):
    cost = 0
    for route in route_list:
        cur_pos = depot - 1
        for job in route:
            cost += cost_graph[cur_pos][job.a]
            cost += job.cost
            cur_pos = job.b
        cost += cost_graph[cur_pos][depot - 1]
    return cost


'''
def init_population(pop_size):
    pop_list = []
    for i in range(pop_size):
        pop_list.append((path_scanning(random.randint(1, 666))))
        # print_route(pop_list[i][0])
        # print("q", pop_list[i][1])
    return pop_list


def select(num, pop_list):
    def compare_fitness(p1, p2):
        return cal_cost(p1) - cal_cost(p2)

    pop_list.sort(key=functools.cmp_to_key(compare_fitness))
    print(cal_cost(pop_list[0]), cal_cost(pop_list[1]))
    return pop_list[:num]
'''


# 小步长定向搜索
def cal_empty_and_cost(route_list):
    res = []
    for route in route_list:
        load_cnt = 0
        cost_cnt = 0
        cur_pos = depot
        for job in route:
            load_cnt += job.demand
            cost_cnt += cost_graph[cur_pos][job.a]
            cost_cnt += cost_graph[job.a][job.b]
            cur_pos = job.b
        res.append((route, capacity - load_cnt, cost_cnt))
    return res


# 计算单个route cost
def cal_single_cost(route):
    cost_cnt = 0
    cur_pos = depot
    for job in route:
        cost_cnt += cost_graph[cur_pos][job.a]
        cost_cnt += job.cost
        cur_pos = job.b
    return cost_cnt


def cal_single_demand(route):
    demand = 0
    for job in route:
        demand += job.demand
    return demand


# 遍历找出所有的job和route对，然后在其中random
def single_insertion(route_list):
    def cmp(e1, e2):
        return e1[1] - e2[1]

    route_empty_and_cost = cal_empty_and_cost(route_list)
    route_empty_and_cost.sort(key=functools.cmp_to_key(cmp))
    job_route_pair = []
    local_demand_list = demand_list.copy()
    local_demand_list.sort()
    last_index = 0
    for job in local_demand_list:
        for i in range(last_index, len(route_empty_and_cost)):
            e = route_empty_and_cost[i]
            if job.demand < e[1]:
                job_route_pair.append((job, route_empty_and_cost[i:]))
                last_index = i
                break
    if len(job_route_pair) == 0: return route_list
    a = random.randint(0, len(job_route_pair) - 1)
    e = job_route_pair[a]
    job = e[0]
    big_list = e[1]
    a = random.randint(0, len(big_list))
    if a == len(big_list):  # 插入自己
        my_route = []
        for route in route_list:
            if job in route:
                my_route = route
                break
        if len(my_route) <= 1: return route_list
        a = random.randint(0, len(my_route) - 1)  # insert position
        b = my_route.index(job)
        if a < b:
            new_route1 = my_route[:a] + [job] + my_route[a:b] + my_route[b + 1:]
            val = cal_single_cost(new_route1)
            new_job = demand_node(job.b, job.a, job.demand, job.cost)
            new_route2 = my_route[:a] + [new_job] + my_route[a:b] + my_route[b + 1:]
            if cal_single_cost(new_route2) < val:
                new_route = new_route2
            else:
                new_route = new_route1
        else:
            new_route1 = my_route[:b] + my_route[b + 1:a] + [job] + my_route[a:]
            val = cal_single_cost(new_route1)
            new_job = demand_node(job.b, job.a, job.demand, job.cost)
            new_route2 = my_route[:b] + my_route[b + 1:a] + [new_job] + my_route[a:]
            if cal_single_cost(new_route2) < val:
                new_route = new_route2
            else:
                new_route = new_route1
        index = route_list.index(my_route)
        route_list[index] = new_route
        return route_list
    else:
        route = big_list[a][0]
        a = random.randint(0, len(route) - 1)  # insert position
        route1 = route[:a] + [job] + route[a:]
        cnt = 0
        cntt = 0
        while cal_single_cost(route) < cal_single_cost(route1):
            cnt += 1
            if cnt > 10:
                a = random.randint(0, len(big_list) - 1)
                route = big_list[a][0]
                cnt = 0
                cntt += 1
            if cntt > 10:
                return route_list
            a = random.randint(0, len(route) - 1)
            route1 = route[:a] + [job] + route[a:]
        i = route_list.index(route)
        route_list[i] = route1
        for r in route_list:
            if job in r:
                index = r.index(job)
                route2 = r[:index] + r[index + 1:]
                i = route_list.index(r)
                route_list[i] = route2
        return route_list


def swap(route_list):
    a = random.randint(0, len(route_list) - 1)
    b = random.randint(0, len(route_list) - 1)
    while a == b:
        b = random.randint(0, len(route_list) - 1)
    route1 = route_list[a]
    route2 = route_list[b]
    demand1 = cal_single_demand(route1)
    demand2 = cal_single_demand(route2)
    c = random.randint(0, len(route1) - 1)
    d = random.randint(0, len(route2) - 1)
    job1 = route1[c]
    job2 = route2[d]
    # 合法性检验
    cnt = 0
    while True:
        if demand1 - job1.demand + job2.demand <= capacity and demand2 - job2.demand + job1.demand <= capacity: break
        cnt += 1
        if cnt > 10: return route_list
        c = random.randint(0, len(route1) - 1)
        d = random.randint(0, len(route2) - 1)
        job1 = route1[c]
        job2 = route2[d]
    val = cal_single_cost(route1) + cal_single_cost(route2)
    cnt = 0
    while True:
        new_route1 = route1[:c] + [job2] + route1[c + 1:]
        new_route2 = route2[:d] + [job1] + route2[d + 1:]
        if val >= cal_single_cost(new_route1) + cal_single_cost(new_route2): break
        cnt += 1
        if cnt > 10: return route_list
        c = random.randint(0, len(route1) - 1)
        d = random.randint(0, len(route2) - 1)
        job1 = route1[c]
        job2 = route2[d]
    route_list[a] = route1
    route_list[b] = route2
    return route_list


def single_reverse(route_list):
    choosable_list = []
    for route in route_list:
        if len(route) > 1:
            choosable_list.append(route)
    if len(choosable_list) == 0: return route_list
    index = random.randint(0, len(choosable_list) - 1)
    route = choosable_list[index]
    val = cal_single_cost(route)
    a = random.randint(0, len(route) - 1)
    b = route[a:]
    new_route = route[:a]
    for job in b[::-1]:
        new_route.append(demand_node(job.b, job.a, job.demand, job.cost))
    '''
    cnt = 0
    while cal_single_cost(new_route) > val:
        cnt += 1
        if cnt > 20: return route_list
        a = random.randint(0, len(route) - 1)
        b = route[a:]
        new_route = route[:a]
        for job in b[::-1]:
            new_route.append(demand_node(job.b, job.a, job.demand, job.cost))
    '''

    index = route_list.index(route)
    route_list[index] = new_route
    return route_list


def double_reverse(route_list):
    choosable_list = []
    for route in route_list:
        if len(route) > 1:
            choosable_list.append(route)
    if len(choosable_list) <= 1: return route_list
    index1 = random.randint(0, len(choosable_list) - 1)
    route1 = choosable_list[index1]
    a1 = random.randint(1, len(route1))
    index2 = random.randint(0, len(choosable_list) - 1)
    while index2 == index1:
        index2 = random.randint(0, len(choosable_list) - 1)
    route2 = choosable_list[index2]
    a2 = random.randint(1, len(route2))


'''
def GA(iteration=1000, population=100):
    pop_list = init_population(population)
    X = []
    Y = []
    for i in range(iteration):
        a = random.randint(0, population - 1)
        # print("flip", cal_cost(pop_list[a]))
        pop_list.append(flip(pop_list[a]))
        a = random.randint(0, population - 1)
        # print("swap", cal_cost(pop_list[a]))
        pop_list.append(swap(pop_list[a]))
        a = random.randint(0, population - 1)
        # print("single reverse", cal_cost(pop_list[a]))
        pop_list.append(single_reverse(pop_list[a]))
        a = random.randint(0, population - 1)
        # print("single insert", cal_cost(pop_list[a]))
        pop_list.append(single_insertion(pop_list[a]))
        pop_list = select(population, pop_list)
        X.append(i)
        Y.append(cal_cost(pop_list[0]))
        # print(cal_cost(pop_list[0]))
    return X, Y
'''


def SA(startNode):
    def expSchedule(k, lam, limit, t):
        if (t > limit):
            return k * math.exp(-lam * t)
        return 0

    X = []
    Y = []
    cnt = 1
    t2 = time.time()
    cur_time = t2 - t1
    rem_time = 1
    x = startNode
    optimal_route = startNode
    optimal_cost = cal_cost(optimal_route)
    # print_route(startNode)
    # print(optimal_cost)
    k = optimal_cost
    lam = 10 * math.e
    limit = 0.01
    ctt=0
    cttt=0
    while (cur_time <= time_limit * 0.95):
        T = expSchedule(k, lam, limit, rem_time)
        # chose method
        if random.random() > 0.3:
            y = single_reverse(x)
            val = cal_cost(y)
            z = swap(x)
            val_z = cal_cost(z)
            if val_z < val:
                y = z
                val = val_z
            r = single_insertion(x)
            val_r = cal_cost(r)
            if val_r < val:
                y = r
                val = val_r
        else:
            y = merge_split(x)
            val = cal_cost(y)
            rem_time = 1
        if val < optimal_cost:
            optimal_route = copy.deepcopy(y)
            optimal_cost = val
        '''
        if T==0 or cal_cost(x) > val:
            pass
        elif (math.exp(cal_cost(x) - val) / T) > 0.5:
            ctt+=1
        else:
            cttt+=1
        '''
        if cal_cost(x) > val or (T != 0 and math.exp(cal_cost(x) - val) / T > random.random()):
            x = y
        X.append(cnt)
        Y.append(val)
        cnt += 1
        t2 = time.time()
        cur_time = t2 - t1
        rem_time *= 0.99
        # print(optimal_cost)
        # rem_time = time_limit - cur_time
    # print(ctt,cttt)
    # print(optimal_cost)
    # print_route(optimal_route)
    return optimal_route, optimal_cost, X, Y


def merge_split(route_list):
    local_route_list = copy.deepcopy(route_list)
    num = random.randint(0, len(route_list) - 1)  # 选几个拆开
    # print("split", num, "routes from", len(route_list), "routes")
    splitted_job_list = []
    for i in range(num):
        a = random.randint(0, len(local_route_list) - 1)  # 要拆开的route下标
        splitted_job_list += local_route_list[a]
        local_route_list.remove(local_route_list[a])
    # print("unchanged route:")
    # print_route(local_route_list)
    local_route_list += path_scanning(splitted_job_list)
    return local_route_list


# 每5s merge_split大跳一次
def search(start_node):
    optimal_route = start_node
    optimal_cost = cal_cost(start_node)
    X = []
    Y = []
    cnt = 1
    while time.time() - t1 < time_limit - 1:
        start_node = merge_split(start_node)
        c = cal_cost(start_node)
        X.append(cnt)
        cnt += 1
        Y.append(c)
        if c < optimal_cost:
            optimal_cost = c
            optimal_route = start_node
    return optimal_route, optimal_cost, X, Y


# ----------read data-------------
file_name = sys.argv[1]
time_limit = int(sys.argv[3])
random_seed = int(sys.argv[5])
random.seed(random_seed)
with open(file_name, encoding='utf-8') as file_obj:
    lines = file_obj.readlines()
num_v = int(lines[1].split(" : ")[1])
depot = int(lines[2].split(" : ")[1])
num_required_edge = int(lines[3].split(" : ")[1])
num_non_edge = int(lines[4].split(" : ")[1])
capacity = int(lines[6].split(" : ")[1])
total_cost = int(lines[7].split(" : ")[1])

cost_graph = np.zeros((num_v, num_v))
demand_list = []
cost_graph[:, :] = np.inf
row, col = np.diag_indices_from(cost_graph)
cost_graph[row, col] = 0
for i in range(num_required_edge + num_non_edge):
    line = lines[9 + i]
    a = int(line.split()[0]) - 1
    b = int(line.split()[1]) - 1
    cost = int(line.split()[2])
    demand = int(line.split()[3])
    cost_graph[a][b] = cost
    cost_graph[b][a] = cost
    if demand != 0: demand_list.append(demand_node(a, b, demand, cost))

# ---------------time----------------
t1 = time.time()

# ---------------floyd----------------
for k in range(num_v):
    for i in range(num_v):
        for j in range(num_v):
            if cost_graph[i][j] > cost_graph[i][k] + cost_graph[j][k]:
                cost_graph[i][j] = cost_graph[i][k] + cost_graph[j][k]

# --------------path scanning-------------
start_node = path_scanning(demand_list)
# print_route(start_node)
# print(cal_cost(start_node))
# optimal_route, optimal_cost, X, Y = search(start_node)
optimal_route, optimal_cost, X, Y = SA(start_node)
# optimal_route = single_reverse(start_node)
print_route(optimal_route)
print("q", int(cal_cost(optimal_route)))
# print("q", int(optimal_cost))

# plt.plot(X, Y)
# plt.show()
