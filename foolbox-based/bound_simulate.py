import numpy as np

B = 100
d = 3*224*224
#w = np.sqrt( 1.0/(5*(d-1)) )
w = 0.001
print ("w:", w)

### Theoretical Part ###
c_worst = ( 2 * ( (1-w**2) ** ((d-1.0)/2.0) ) - 1)
best_l = (2.0/np.pi)*np.sqrt(1.0*B/d)
best_u = np.sqrt(1.0*B/d)
worst_l = c_worst*(2.0/np.pi)*np.sqrt(1.0*B/d)
worst_u = c_worst*np.sqrt(1.0*B/d)
print ("Best case, expectation in (%.6f, %.6f)"%(best_l, best_u))
print ("Worst case, expectation in (%.6f, %.6f)"%(worst_l, worst_u))
#print ("Best case, expectation in (%.6f, %.6f)"%(np.sqrt(0.5*B/d), np.sqrt(2.0*B/d)))
#print ("Worst case, expectation in (%.6f, %.6f)"%(c_worst*np.sqrt(0.5*B/d), c_worst*np.sqrt(2.0*B/d)))

### Exp Part ###
N_trial = 100
bests = []
worsts = []
for _ in range(N_trial):
    print (_)
    grad = np.random.randn(d)
    grad = grad / np.linalg.norm(grad)
    grad = np.abs(grad)
    cos = grad[:B].sum() / (np.sqrt(B))
    bests.append(cos)

    grad[grad<w] = -grad[grad<w]
    cos = grad[:B].sum() / (np.sqrt(B))
    worsts.append(cos)
    print ("Trial %d, Best case: %.6f, Worst case: %.6f"%(_, bests[-1], worsts[-1]))

import matplotlib.pyplot as plt
fig = plt.figure()
x = [1.0, 2.0]
width=0.3
height = [best_u-best_l, worst_u-worst_l]
bottom = [best_l, worst_l]
plt.bar(x, height, bottom=bottom, width=width, label='Theoretical Expectation')
plt.plot([x[0]]*len(bests)+[x[1]]*len(worsts), bests+worsts, 'r.', label='Stimulated Value')
plt.ylim(0.0125, 0.0275)
plt.xlim(0,3)
plt.xticks(np.array(x), ('Upper Bound', 'Lower Bound'))
plt.ylabel('Cosine Similarity')
plt.legend()
plt.show()
fig.savefig('stimulated.pdf')
