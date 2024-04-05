import numpy as np
import matplotlib.pyplot as plt
from pylab import xticks, yticks
from scipy.optimize import linprog

##env###
S = 100
A1, A2, A3 = 4, 3, 2

np.random.seed(1)

P = np.random.uniform(0,1,size=(S,A1,A2,A3,S))  #action-independent transition
for t in range(S//2):
    i = np.random.randint(S)
    P[:,:,:,:,i] = np.random.uniform(0,0.1,size=(S,A1,A2,A3))
P = P*((1/P.sum(-1)).reshape(S,A1,A2,A3,1).dot(np.ones((1,S))))

R = np.random.uniform(0,1,size=(S,A1,A2,A3))
# for i in range(S):
#   j1, j2, j3 = np.random.randint(A1), np.random.randint(A2), np.random.randint(A3)
#   R[i,j1,j2,j3] = np.random.uniform(0.6,1)
##end env#####


def projection_simplex_sort(v, z=1):
    ## reference from https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    # input 1-dim vector v, project into simplex
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def proximal_d1(g, p):
    # solve argmax_x\in_simplex {<g,x>-1/2||x-p||_2^2}
    # g,p is 1-dim vector
    index = np.argsort(g+p)[::-1]   # ascending index
    sum = g[index[0]]+p[index[0]]
    maxOf0 = None  #maxOf0 = max_{i\in_I0} g_i+p_i
    piv = 0   #I+ = [0:piv],  I0 = [piv:-1], sum = sum_{i\in_I+} g_i+p_i
    candidate = []
    for i in range(1,index.shape[0]):
        if g[index[i-1]]+p[index[i-1]]==g[index[i]]+p[index[i]]:
            sum += g[index[i]]+p[index[i]]
            continue
        piv = i
        maxOf0 = g[index[i]]+p[index[i]]
        if (sum-1)/piv<g[index[i-1]]+p[index[i-1]] and (sum-1)/piv>=maxOf0:
            candidate.append((piv,sum))
        sum += g[index[i]]+p[index[i]]

    if not maxOf0:
        return [np.ones(g.shape[0])/g.shape[0]]

    # for piv==n
    piv = g.shape[0]
    if (sum-1)/piv<g[index[piv-1]]+p[index[piv-1]]:
        candidate.append((piv,sum))

    res = []
    for i,s in candidate:
        c = (s-1)/i
        x = np.zeros(g.shape[0])
        x[index[:i]] = p[index[:i]]+g[index[:i]]-c
        res.append(x)

    return res

def proximalQ(g,p):
    res = np.zeros(p.shape)
    for s in range(p.shape[0]):
        # res[s,:] = proximal_d1(g[s,:],p[s,:])[0]
        res[s,:] = projection_simplex_sort(g[s,:]+p[s,:])
    return res

def oracle_P(p1,p2,p3):
    temp = np.zeros((S,S))
    for s in range(S):
        for a1 in range(A1):
            for a2 in range(A2):
                for a3 in range(A3):
                    temp[s,:] += P[s,a1,a2,a3,:]*p1[s,a1]*p2[s,a2]*p3[s,a3]
    return temp

def oracle_SD(p1, p2, p3):
    P_pi = oracle_P(p1,p2,p3)
    a = np.concatenate((np.diag(np.ones(S)) - P_pi.T,  np.ones((1,S))),axis=0)
    b = np.array([0]*S+[1])
    x = np.linalg.lstsq(a,b,rcond=None)
    return x[0]


def oracle_Q(policy1, policy2, policy3):
    nu = oracle_SD(policy1, policy2, policy3)
    RofS = (policy1.T*(R.T)).sum(axis=2)  #sum over A1
    RofS = (policy2.T*(RofS)).sum(axis=1)  #sum over A2
    RofS = (policy3.T*(RofS)).sum(axis=0)  #sum over A3

    rho = nu.dot(RofS)

    P_pi = oracle_P(policy1, policy2, policy3)
    a = np.concatenate((np.diag(np.ones(S)) - P_pi.T, nu.reshape(1,-1)) ,axis=0)
    b = np.concatenate( (RofS - rho*np.ones(S), np.array([0])) )
    V = np.linalg.lstsq(a,b,rcond=None)[0]
    Q = R - rho*np.ones((S,A1,A2,A3)) + P.dot(V)

    x = (policy1.T*(Q.T)).sum(axis=2)
    y3 = (policy2.T*(x)).sum(axis=1)

    RR = np.transpose((Q.T),(1,0,2,3))
    x = (policy1.T*(RR)).sum(axis=2)
    y2 = (policy3.T*(x)).sum(axis=1)

    RR = np.transpose((Q.T),(2,1,0,3))
    x = (policy3.T*(RR)).sum(axis=2)
    y1 = (policy2.T*(x)).sum(axis=1)

    return [y1.T, y2.T, y3.T]


def proj(policy):
    res = np.zeros(policy.shape)
    for i in range(policy.shape[0]):
        res[i] = projection_simplex_sort(policy[i])
    return res

def solveAMDP(P,R):
    # P:S*A*S, R:S*A
    # variable for opt is x=[g,h]
    def rosen(x):
        """The Rosenbrock function with additional arguments"""
        return x[0]
    s, a = P.shape[0], P.shape[1]
    A_h = P.reshape(-1,s)-np.repeat(np.eye(s),a,axis=0)
    A = np.concatenate((-np.ones((s*a,1)),A_h),axis=1)
    ub = -R.reshape(-1)

    c = np.zeros(s+1)
    c[0] = 1
    res = linprog(c, A_ub=A, b_ub=ub)
    h = res.x[1:]

    policy = np.zeros((s,a))
    policy[np.arange(s),np.argmax(R+P.dot(h), axis=1)] = 1
    P_pi, R_pi = np.zeros((s,s)), np.zeros(s)
    for s1 in range(s):
        for a1 in range(a):
            P_pi[s1,:] += P[s1,a1,:]*policy[s1,a1]
            R_pi[s1] += R[s1,a1]*policy[s1,a1]
    a = np.concatenate((np.diag(np.ones(s)) - P_pi.T,  np.ones((1,s))),axis=0)
    b = np.array([0]*s+[1])
    x = np.linalg.lstsq(a,b,rcond=None)

    return x[0].dot(R_pi)

def NashGap(p1, p2, p3):
    P1, P2, P3 = np.zeros((S,A1,S)),np.zeros((S,A2,S)),np.zeros((S,A3,S))
    R1, R2, R3 = np.zeros((S,A1)), np.zeros((S,A2)), np.zeros((S,A3))
    for s in range(S):
        for a1 in range(A1):
            for a2 in range(A2):
                for a3 in range(A3):
                    P1[s,a1,:] += P[s,a1,a2,a3,:]*p2[s,a2]*p3[s,a3]
                    P2[s,a2,:] += P[s,a1,a2,a3,:]*p1[s,a1]*p3[s,a3]
                    P3[s,a3,:] += P[s,a1,a2,a3,:]*p1[s,a1]*p2[s,a2]
                    R1[s,a1] += R[s,a1,a2,a3]*p2[s,a2]*p3[s,a3]
                    R2[s,a2] += R[s,a1,a2,a3]*p1[s,a1]*p3[s,a3]
                    R3[s,a3] += R[s,a1,a2,a3]*p1[s,a1]*p2[s,a2]
    res = [solveAMDP(P1,R1),solveAMDP(P2,R2),solveAMDP(P3,R3)]

    nu = oracle_SD(p1, p2, p3)
    RofS = (p1.T*(R.T)).sum(axis=2)  #sum over A1
    RofS = (p2.T*(RofS)).sum(axis=1)  #sum over A2
    RofS = (p3.T*(RofS)).sum(axis=0)  #sum over A3

    rho = nu.dot(RofS)
    return max(res)-rho

def projPG(T, lr):
    policy1, policy2, policy3 = np.ones((S,A1))/A1, np.ones((S,A2))/A2, np.ones((S,A3))/A3
    # policy1, policy2, policy3 = pt[-1]
    policyTraj = [ [policy1, policy2, policy3] ]
    Nash_gaps = [NashGap(policy1, policy2, policy3)]

    statDist = oracle_SD(policy1, policy2, policy3)
    for i in range(T):
        Q1, Q2, Q3 = oracle_Q(policy1, policy2, policy3)
        policy1 = proj(policy1 + lr * np.diag(statDist).dot(Q1))
        policy2 = proj(policy2 + lr * np.diag(statDist).dot(Q2))
        policy3 = proj(policy3 + lr * np.diag(statDist).dot(Q3))
        # print(np.diag(statDist).dot(Q1))

        policyTraj.append([policy1, policy2, policy3])
        Nash_gaps.append(NashGap(policy1, policy2, policy3))

    return policyTraj, Nash_gaps

def proxQ(T, lr):
    policy1, policy2, policy3 = np.ones((S,A1))/A1, np.ones((S,A2))/A2, np.ones((S,A3))/A3
    policyTraj = [ [policy1, policy2, policy3] ]
    Nash_gaps = [NashGap(policy1, policy2, policy3)]

    for i in range(T):
        Q1, Q2, Q3 = oracle_Q(policy1, policy2, policy3)
        policy1 = proximalQ(lr*Q1, policy1)
        policy2 = proximalQ(lr*Q2, policy2)
        policy3 = proximalQ(lr*Q3, policy3)

        policyTraj.append([policy1, policy2, policy3])
        Nash_gaps.append(NashGap(policy1, policy2, policy3))

    return policyTraj, Nash_gaps

def explore_c(policy1, policy2, policy3):
    Q1, Q2, Q3= oracle_Q(policy1, policy2,policy3)
    bestaction1, bestaction2, bestaction3 = Q1.argmax(axis=1), Q2.argmax(axis=1), Q3.argmax(axis=1)
    S = policy1.shape[0]
    c1 = policy1[np.arange(S), bestaction1].min()
    c2 = policy2[np.arange(S), bestaction2].min()
    c3 = policy3[np.arange(S), bestaction3].min()

    return min([c1,c2,c3])

def NPG(T, lr):
    policy1, policy2, policy3 = np.ones((S,A1))/A1, np.ones((S,A2))/A2, np.ones((S,A3))/A3
    policyTraj = [[np.copy(policy1), np.copy(policy2), np.copy(policy3)] ]
    Nash_gaps = [NashGap(policy1, policy2,policy3)]
    c_traj = [explore_c(policy1, policy2,policy3)]

    for i in range(T):
        Q1, Q2, Q3= oracle_Q(policy1, policy2,policy3)
        for s in range(S):
            policy1[s,:] = policy1[s,:]*np.exp(lr*Q1[s,:])
            policy1[s,:] = policy1[s,:]/policy1[s,:].sum()
            policy2[s,:] = policy2[s,:]*np.exp(lr*Q2[s,:])
            policy2[s,:] = policy2[s,:]/policy2[s,:].sum()
            policy3[s,:] = policy3[s,:]*np.exp(lr*Q3[s,:])
            policy3[s,:] = policy3[s,:]/policy3[s,:].sum()

        policyTraj.append([np.copy(policy1), np.copy(policy2), np.copy(policy3)])
        Nash_gaps.append(NashGap(policy1, policy2, policy3))
        c_traj.append(explore_c(policy1, policy2,policy3))

    return policyTraj, Nash_gaps, c_traj

def main():
    pt, ngs = projPG(500,400)
    pt2, ngs2 = proxQ(500,4)
    pt3, ngs3, c_traj = NPG(500,1)

    logNG1, logNG2, logNG3 = np.log(ngs), np.log(ngs2), np.log(ngs3)
    for i in range(len(logNG1)):
        if logNG1[i]<-35:
            logNG1[i]=-35
    for i in range(len(logNG2)):
        if logNG2[i]<-35:
            logNG2[i]=-35
    for i in range(len(logNG3)):
        if logNG3[i]<-35:
            logNG3[i]=-35

    index = 500
    plt.plot([i for i in range(index)], logNG1[:index], label='PG',color='darkred', alpha=0.75,linewidth=2)
    plt.plot([i for i in range(index)], logNG2[:index], label='ProxQ',color='darkblue', alpha=0.75,linewidth=2)
    plt.plot([i for i in range(index)], logNG3[:index], label='NPG',color='darkgoldenrod', alpha=0.75,linewidth=2)
    # plt.plot([i+1 for i in range(len(ngs3))], ngs3, label='NPG',color='darkgoldenrod', alpha=0.75)
    plt.legend(bbox_to_anchor=(1,1), loc="upper right", borderaxespad=0, fontsize="16")
    xticks(np.linspace(0,500,6,endpoint=True),size=15)
    yticks(np.linspace(-35,-5,7,endpoint=True),size=15)
    plt.grid(axis='x',linestyle='-.',linewidth=0.5,color='black',alpha=0.5)
    plt.grid(axis='y',linestyle='-.',linewidth=0.5,color='black',alpha=0.5)
    ax = plt.gca()
    ax.set_xlim(left=0)

    plt.xlabel('iteration', size=20)
    plt.ylabel('log(Nash gap)', size=20)


if __name__=="__main__":
    main()
