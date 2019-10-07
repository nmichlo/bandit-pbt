
import numpy as np
from tqdm import tqdm

k = 4
r = 10

step_dists = np.array([[0 if i == j else np.inf for j in range(k)] for i in range(k)])
expl_dists = np.array([[0 if i == j else np.inf for j in range(k)] for i in range(k)])

step_count = np.zeros(k, dtype='int')
expl_count = np.zeros(k, dtype='int')

parent_hist = [['|']*k]

step_count_hist = [step_count.copy()]
expl_count_hist = [expl_count.copy()]

step_dist_hist = [step_dists.copy()]
expl_dist_hist = [expl_dists.copy()]

def ran_index():
    return np.random.randint(0, k)

def exploit(i):
    r = i
    while r == i:
        r =  ran_index()
    return r


# replace_exploit = [(i, exploit(i)) for i in (ran_index() for _ in range(r))]
replace_exploit = [
    (0, 1),
    (2, 0),
    (3, 2),
    (0, 1),
    (2, 3),
]

for idx_replace, idx_exploit in replace_exploit:
    step_count += 1
    step_count[idx_replace] = 0

    expl_count[idx_replace] = 0
    expl_count[idx_exploit] += 1

    parents = parent_hist[-1][:]
    parents[idx_replace] = idx_exploit

    # STEP DISTS
    step_dists[idx_replace, :] = step_dists[idx_exploit, :]
    step_dists[:, idx_replace] = step_dists[:, idx_exploit]
    step_dists += 2
    np.fill_diagonal(step_dists, 0)
    step_dists[idx_replace, idx_exploit] = 0
    step_dists[idx_exploit, idx_replace] = 0


    # EXPL DISTS
    # expl_dists[idx_replace, :] = expl_dists[idx_exploit, :]
    # expl_dists[:, idx_replace] = expl_dists[:, idx_exploit]

    # exploit_mask = np.zeros((k, k), bool)
    # exploit_mask[idx_exploit, :] = True
    # exploit_mask[:, idx_exploit] = True

    match = parents == parent_hist[-1]

    incr_map = np.ones((k, k), bool)
    incr_map[:, match] = False
    incr_map[match, :] = False

    expl_dists[:, idx_replace] = expl_dists[:, idx_exploit]
    expl_dists[idx_replace, :] = expl_dists[idx_exploit, :]
    expl_dists[incr_map] += 1

    np.fill_diagonal(expl_dists, 0)

    # HISTORIES
    parent_hist.append(parents)
    step_count_hist.append(step_count.copy())
    expl_count_hist.append(expl_count.copy())
    step_dist_hist.append(step_dists.copy())
    expl_dist_hist.append(expl_dists.copy())

tqdm.write('\nPARENTS:')
for parents in parent_hist[1:]:
    tqdm.write(' '.join(f'{p}' for p in parents)[::-1])

tqdm.write('\nEXPLOIT COUNTS:')
for expl_count in expl_count_hist[1:]:
    tqdm.write(' '.join(f'{p}' for p in expl_count)[::-1])

tqdm.write('\nSTEP COUNTS:')
for step_count in step_count_hist[1:]:
    tqdm.write(' '.join(f'{p}' for p in step_count)[::-1])

i = 2

tqdm.write(f'\nSTEP DISTS [{i}]:')
for step_dists in step_dist_hist[1:]:
    tqdm.write(
        '  |||  '.join(' '.join(f'{p:3.0f}'[::-1] for p in step_dists[i])[::-1] for i in range(k))
    )


tqdm.write(f'\nEXPL DISTS [{i}]:')
for expl_dists in expl_dist_hist[1:]:
    tqdm.write(
        '  |||  '.join(' '.join(f'{p:3.0f}'[::-1] for p in expl_dists[i])[::-1] for i in range(k))
    )


# for prv, nxt, steps in zip(parent_hist[:-1], parent_hist[1:], step_dist_hist[:-1]):
#     for p, n in zip(prv, nxt):
#         tqdm.write(p, end=' ')
#     tqdm.write('')
# tqdm.write(f'{np.array(parent_hist)}')


