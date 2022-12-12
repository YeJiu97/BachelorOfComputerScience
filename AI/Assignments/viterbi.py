import sys
import numpy as np

OBSTACLE = 'X'
EMPTY = '0'


def load_data(filename):
    """Given the input filename, read map, sensor observation and error rate"""
    with open(filename, 'r') as f:
        # The first line indicates the size of the map (rows by columns).
        line = f.readline().strip()
        rows, columns = list(map(int, line.split()))
        # read the map
        map_data = []
        for i in range(rows):
            line = f.readline().strip()
            map_data.append(line.split())
        map_data = np.array(map_data)
        #  Next line specifies the number of sensor observation(s).
        line = f.readline().strip()
        n = int(line)
        # The observed values then follows for the specified number of rows corresponding
        # to each time step.
        observations = []
        for i in range(n):
            line = f.readline().strip()
            observations.append(line)
        # The last line specifies the sensor’s error rate ε
        line = f.readline().strip()
        err = float(line)
    return map_data, observations, err


def viterbi(map_data, Y, err):
    """
    Viterbi algorithm
    map_data: numpy array, the map data
    Y: the observed sequence
    err: sensor’s error rate
    """
    # generate observation space: '0000', '0001', ..., '1111'
    O = []
    for i in range(16):
        binary_str = bin(i)[2:]
        # pad '0'
        binary_str = '0' * (4 - len(binary_str)) + binary_str
        O.append(binary_str)
    rows, cols = map_data.shape
    # build the state space, each element is a tuple like (r, c) represent
    # the traversable positions
    S = []
    for r in range(rows):
        for c in range(cols):
            if map_data[r][c] == EMPTY:
                S.append((r, c))
    K = len(S)
    N = len(O)
    # initialize the transition matrix and emission matrix
    Tm = np.zeros((K, K))
    Em = np.zeros((K, N))  # 16 different status
    for r, c in S:
        cur = (r, c)
        observed = ''
        # neighbours,  set of traversable points that are adjacent to
        neighbours = []
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):  # NESW
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if map_data[nr][nc] == EMPTY:
                    observed += '0'
                    neighbours.append((nr, nc))
                else:  # blocked
                    observed += '1'
            # invalid place
            else:
                observed += '1'
        i = S.index(cur)
        # size
        ni = len(neighbours)
        if ni > 0:
            # has equal probability to move to each of its neighbours.
            p = 1 / ni
            for neighbour in neighbours:
                j = S.index(neighbour)
                Tm[i][j] = p
        # calculate the emission matrix
        for j, observation in enumerate(O):
            # count the number of different bits
            d = sum(observation[k] != observed[k] for k in range(4))
            p = (1 - err) ** (4 - d) * (err ** d)
            Em[i][j] = p
    initial_probabilities = [1 / K for _ in range(K)]
    T = len(Y)
    trellis = np.zeros((K, T))
    for i in range(K):
        y1 = O.index(Y[0])
        trellis[i, 0] = initial_probabilities[i] * Em[i, y1]
    for j in range(1, T):
        for i in range(K):
            yj = O.index(Y[j])
            trellis[i, j] = max(trellis[k, j - 1] * Tm[k, i] * Em[i, yj] for k in range(K))
    return S, trellis


def main():
    if len(sys.argv) != 2:
        print(sys.argv)
        print('Usage: python viterbi.py [input]')
        exit(-1)
    filename = sys.argv[1]
    map_data, Y, err = load_data(filename)
    S, trellis = viterbi(map_data, Y, err)
    maps = []
    for i in range(len(Y)):
        m = np.zeros(map_data.shape)
        probs = trellis[:, i]
        for j, (r, c) in enumerate(S):
            m[r][c] = probs[j]
        maps.append(m)
    np.savez('output.npz', *maps)


if __name__ == '__main__':
    main()
