import sys
from queue import Queue, PriorityQueue

# up, down, left, right
drc = ((-1, 0), (1, 0), (0, -1), (0, 1))


def load_map(filename):
    with open(filename, 'r') as f:
        rows, columns = list(map(int, f.readline().split()))
        start_r, start_c = list(map(int, f.readline().split()))
        end_r, end_c = list(map(int, f.readline().split()))
        map_data = []
        for i in range(rows):
            line = f.readline()
            line = line.split()
            row = []
            for e in line:
                if e == 'X':
                    row.append(e)
                else:
                    row.append(int(e))
            map_data.append(row)
    return start_r - 1, start_c - 1, end_r - 1, end_c - 1, map_data


def print_solution(solution, map_data):
    if not solution:
        print('null')
        return
    cols = len(map_data[0])
    for r, row in enumerate(map_data):
        for c, col in enumerate(row):
            if (r, c) in solution:
                print('*', end='')
            else:
                print(col, end='')
            if c != cols - 1:
                print(' ', end='')
            else:
                print()


def bfs(start_r, start_c, end_r, end_c, map_data):
    rows = len(map_data)
    cols = len(map_data[0])
    queue = Queue()
    visited = set()
    parent = {(start_r, start_c): None}
    queue.put((start_r, start_c, [(start_r, start_c)]))
    solution = []
    while not queue.empty():
        r, c, path = queue.get()
        if (r, c) in visited:
            continue
        if (r, c) == (end_r, end_c):
            solution = path
            break
        visited.add((r, c))
        for dr, dc in drc:
            nr, nc = r + dr, c + dc
            # it's an valid position, and is not obstacle and has not been visited
            if 0 <= nr < rows and 0 <= nc < cols and \
                    map_data[nr][nc] != 'X' and \
                    (nr, nc) not in visited:
                parent[(nr, nc)] = (r, c)
                queue.put((nr, nc, path + [(nr, nc)]))
    print_solution(solution, map_data)


def ucs(start_r, start_c, end_r, end_c, map_data):
    rows = len(map_data)
    cols = len(map_data[0])
    queue = PriorityQueue()
    visited = set()
    parent = {(start_r, start_c): None}
    node_index = 0
    queue.put((0, node_index, [(start_r, start_c)]))
    solution = []
    node_index += 1

    while not queue.empty():
        g, _, path = queue.get()
        r, c = path[-1]
        if (r, c) in visited:
            continue
        if (r, c) == (end_r, end_c):
            solution = path
            break
        visited.add((r, c))
        for dr, dc in drc:
            nr, nc = r + dr, c + dc
            # it's an valid position, and is not obstacle and has not been visited
            if 0 <= nr < rows and 0 <= nc < cols and \
                    map_data[nr][nc] != 'X' and \
                    (nr, nc) not in visited:
                cost = 0
                if map_data[nr][nc] > map_data[r][c]:
                    cost = 1 + map_data[nr][nc] - map_data[r][c]
                else:
                    cost = 1
                parent[(nr, nc)] = (r, c)
                queue.put((g + cost, node_index, path + [(nr, nc)]))
                node_index += 1
    print_solution(solution, map_data)


def manhattan(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)


def euclidean(r1, c1, r2, c2):
    return ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5


def astar(start_r, start_c, end_r, end_c, map_data, heuristic):
    rows = len(map_data)
    cols = len(map_data[0])
    queue = PriorityQueue()
    closed = set()
    costs = {(start_r, start_c): 0}
    parent = {(start_r, start_c): None}
    node_index = 0
    queue.put((0, 0, node_index, [(start_r, start_c)]))
    open = {(start_r, start_c)}
    solution = []
    node_index += 1
    while not queue.empty():
        f, g, _, path = queue.get()
        r, c = path[-1]
        if (r, c) in closed:
            continue
        if (r, c) == (end_r, end_c):
            solution = path
            break
        closed.add((r, c))
        for dr, dc in drc:
            nr, nc = r + dr, c + dc
            # it's an valid position, and is not obstacle
            if 0 <= nr < rows and 0 <= nc < cols and \
                    map_data[nr][nc] != 'X':
                cost = 0
                if map_data[nr][nc] > map_data[r][c]:
                    cost = 1 + map_data[nr][nc] - map_data[r][c]
                else:
                    cost = 1
                cost = g + cost
                if (nr, nc) in closed:
                    continue
                h = cost + heuristic(nr, nc, end_r, end_c)
                if (nr, nc) not in open:
                    queue.put((h, cost, node_index, path + [(nr, nc)]))
                    costs[(nr, nc)] = cost
                    parent[(nr, nc)] = (r, c)
                    node_index += 1
                elif costs[(nr, nc)] < cost:
                    parent[(nr, nc)] = (r, c)
                    costs[(nr, nc)] = cost

    print_solution(solution, map_data)


def main():
    argv = sys.argv
    filename = sys.argv[1]
    algorithm = sys.argv[2]

    start_r, start_c, end_r, end_c, map_data = load_map(filename)
    if algorithm == 'bfs':
        bfs(start_r, start_c, end_r, end_c, map_data)
    elif algorithm == 'ucs':
        ucs(start_r, start_c, end_r, end_c, map_data)
    else:
        # astar
        heuristic = sys.argv[3]
        if heuristic == 'manhattan':
            heuristic = manhattan
        else:
            heuristic = euclidean
        astar(start_r, start_c, end_r, end_c, map_data, heuristic)


if __name__ == '__main__':
    main()

"""
in1.txt bfs
in2.txt ucs
in3.txt astar manhattan
in4.txt astar euclidean
in5.txt astar manhattan
in6.txt astar euclidean
"""
