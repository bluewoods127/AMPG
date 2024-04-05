import numpy as np
import matplotlib.pyplot as plt
from pylab import xticks, yticks

class ENV:
    def __init__(self):
        self.S = 2
        self.A1, self.A2 = 2, 2
        self.P = np.array([[0.9,0.1],[0.3,0.7]])
        self.R = np.array([[[1,0.2],[0.8,0.2]],[[0.2,1],[0.1,0.6]]])
        self.curState = 0

    def getState(self):
        return self.curState

    def step(self,a):
        r = self.R[self.getState(),a[0],a[1]]
        self.curState = np.random.choice(self.S, 1, p=self.P[self.curState,:])[0]
        return r, self.curState

    def oracle_SD(self):
        a = np.concatenate((np.diag(np.ones(self.S)) - self.P.T,  np.ones((1,self.S))),axis=0)
        b = np.array([0]*self.S+[1])
        x = np.linalg.lstsq(a,b,rcond=None)
        return x[0]

    def oracle_Q(self, policy1, policy2):
        nu = self.oracle_SD()
        RofS = (policy1.T*(self.R.T)).sum(axis=1)  #sum over A1
        RofS = (policy2.T*(RofS)).sum(axis=0)  #sum over A2

        rho = nu.dot(RofS)

        a = np.concatenate((np.diag(np.ones(self.S)) - self.P.T, nu.reshape(1,-1)) ,axis=0)
        b = np.concatenate( (RofS - rho*np.ones(self.S), np.array([0])) )
        V = np.linalg.lstsq(a,b,rcond=None)[0]
        Q = self.R - rho*np.ones((self.S,self.A1,self.A2)) + np.repeat(self.P.dot(V.reshape(-1,1)).reshape(-1,1), self.A1*self.A2, axis=1).reshape(self.S,self.A1,self.A2)

        y2 = (policy1.T*(Q.T)).sum(axis=1)

        RR = np.transpose((Q.T),(1,0,2))
        y1 = (policy2.T*(RR)).sum(axis=1)

        return [y1.T, y2.T]

    def NashGap(self, p1, p2):
        nu = self.oracle_SD()

        x = (p1.T*(self.R.T)).sum(axis=1)
        y = (p2.T*(x)).sum(axis=0)
        r2 = (x.max(axis=0)-y).max()

        RR = np.transpose((self.R.T),(1,0,2))
        x = (p2.T*(RR)).sum(axis=1)
        y = (p1.T*(x)).sum(axis=0)
        r1 = (x.max(axis=0)-y).max()

        return max([r1,r2])

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

def proj(policy):
    res = np.zeros(policy.shape)
    for i in range(policy.shape[0]):
        res[i] = projection_simplex_sort(policy[i])
    return res

def gradEstimate(traj, policy, K, N1, N2):
    rho = 0
    for t in range(N1 // 2, N1):
        rho += traj[t][2]
    rho = rho / (N1 - N1 // 2)

    g = np.zeros(policy.shape)
    for k in range(K):
        gradPi = np.zeros(policy.shape)
        gradPi[traj[N1 + k * N2][0], traj[N1 + k * N2][1]] = 1 / policy[traj[N1 + k * N2][0], traj[N1 + k * N2][1]]
        R = 0
        for tau in range(N1 + k * N2, N1 + (k + 1) * N2):
            R += traj[tau][2] - rho
        g = g + R * np.copy(gradPi)

    return g / K

def projPGSample(env, T, lr, N1=500, N2=20, K=100, alpha=0.05):
    s0 = env.getState()

    policy1, policy2 = np.ones((env.S,env.A1))/env.A1, np.ones((env.S,env.A2))/env.A2
    policyTraj = [ [policy1, policy2] ]
    Nash_gaps = [env.NashGap(policy1, policy2)]

    for i in range(T):
        print(i)
        if i==20:
            lr *= 0.2
        elif i==40:
            lr *= 0.2
        elif i==60:
            lr *= 0.5
        elif i==80:
            lr *= 0.5
        elif i==100:
            lr = 0.0001

        traj1, traj2 = [], []
        for tau in range(N1+K*N2):
            a1 = np.random.choice(env.A1, 1, p=policy1[s0,:])[0]
            a2 = np.random.choice(env.A2, 1, p=policy2[s0,:])[0]

            r, s1 = env.step([a1,a2])
            traj1.append((s0,a1,r))
            traj2.append((s0,a2,r))
            s0 = s1

        grad1 = gradEstimate(traj1, policy1, K, N1, N2)
        grad2 = gradEstimate(traj2, policy2, K, N1, N2)
        policy1 = proj(policy1 + lr * grad1)
        policy2 = proj(policy2 + lr * grad2)
        for s in range(env.S):
            if policy1[s].min()<alpha:
                policy1[s] = (1-alpha)*policy1[s] + alpha*np.ones((env.A1))/env.A1
            if policy2[s].min()<alpha:
                policy2[s] = (1-alpha)*policy2[s] + alpha*np.ones((env.A2))/env.A2

        policyTraj.append([policy1, policy2])
        Nash_gaps.append(env.NashGap(policy1, policy2))

    return policyTraj, Nash_gaps


def main():
    env = ENV()
    np.random.seed(0)
    pt, ng = projPGSample(env, 100, 0.5, 1000, 50, 1000, 0)
    plt.plot([i + 1 for i in range(len(ng))], ng)
    plt.legend(loc="upper right")
    xticks(np.linspace(0, 100, 6, endpoint=True))
    yticks(np.linspace(0, 0.4, 5, endpoint=True))
    plt.grid(axis='x', linestyle='-.', linewidth=0.5, color='black', alpha=0.5)
    plt.grid(axis='y', linestyle='-.', linewidth=0.5, color='black', alpha=0.5)
    ax = plt.gca()
    ax.set_xlim(left=0)

    plt.xlabel('iteration', size=15)
    plt.ylabel('Nash gap', size=15)

if __name__=="__main__":
    main()
