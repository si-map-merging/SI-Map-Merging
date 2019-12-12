"""
Implementation of algorithm 2 in TEASER paper
"""

import numpy as np


def f(s,a,s_hat,c):
    K=len(s)
    objective = 0.
    for i in range(K):
        objective+=min((s_hat-s[i])**2/a[i]**2,c)
    return objective


def adaptive_voting(s,a,c):
    """Algorithm 2 of TEASER paper

    Args:
        s_k, a_k, \bar{c}
    Return:
        \hat{s} (Estimation of scale s)
    """
    # Define boundaries and sort
    K = len(s)
    v=[]

    for i in range(len(s)):
        v.append(s[i]-a[i]*c)
        v.append(s[i]+a[i]*c)
    v=sorted(v)

    # Compute middle points
    m=[]
    for i in range(2*K-1):
        m.append((v[i]+v[i+1])/2.)

    # voting
    S=[]
    for i in range(2*K-1):
        S.append([])

        for k in range(K):

            if m[i]<= s[k] + a[k]*c and m[i]>= s[k] - a[k]*c:
                S[i].append(k)

    # Enumerate consensus sets and return best
    f_list=[]
    for i in range(2*K-1):
        if S[i]:
            ww=0.
            ss = 0.
            for k in S[i]:
                ww+=1./(a[k]**2)
                ss+=s[k]/(a[k]**2)
            s_est = ss/ww
            f_list.append((s_est,f(s,a,s_est,c)))
        else:
            s_est = m[i]
            f_list.append((s_est,f(s,a,s_est,c)))

    f_list  = sorted(f_list, key=lambda s:s[1])
    return f_list[0][0]


if __name__ == '__main__':
    s = [1,2,2.8,3,3.2,4,5,10]
    a = [1.1,1.1,1.3,1.3,1.1,1.1,1.1,1]
    print(adaptive_voting(s,a,1))