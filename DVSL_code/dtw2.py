import pandas as pd
import numpy as np
from scipy.spatial import distance
def dtw_fast(t,r,flag):
    d=distance.cdist(np.transpose(t),np.transpose(r),'sqeuclidean')
    rows,N = t.shape
    rows,M = r.shape
    #
    # print(N)
    # print(M)
    D=np.zeros((d.shape[0],d.shape[1]))
    # print(D.shape)
    D[0][0]=d[0][0]

    for n in range(1,N):
        D[n][0] = d[n][0] + D[n-1][0]

    for m in range(1,M):
        D[0][m] = d[0][m] + D[0][m-1]

    for n in range(1,N):
        for m in range(1,M):
            D[n][m] = d[n][m] + min(D[n-1][m],D[n-1][m-1],D[n][m-1])

    Dist = D[N-1][M-1]

    if flag == True:
        return Dist


    n=N
    m=M
    k=0
    w=[]

    w=np.array([N,M])
    # print(rows)

    while (n+m)!=2:
        if (n-1)==0:
            # print('IN IF')
            m=m-1
        elif (m-1)==0:
            n=n-1
        # else:
        #     else
        #     [values, number] = min([D(n - 1, m), D(n, m - 1), D(n - 1, m - 1)]);
        #     switch
        #     number
        #     case 1
        #         n = n - 1;
        #     case 2
        #         m = m - 1;
        #     case 3
        #         n = n - 1;
        #         m = m - 1;
        k=k+1
        w=np.vstack((w, [n,m]))


    T = np.zeros((N,M))
    # print(T.shape)
    for temp_t in range(0,w.shape[0]):
        # print(w[temp_t][0])
        # print(w[temp_t][1])
    # #     print(temp_t)
        T[(w[temp_t][0])-1][(w[temp_t][1])-1] = 1
    #     print('&&&&&&&&&&&&&&&&&&&&&&&&&')
    # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    # #

    return Dist,T