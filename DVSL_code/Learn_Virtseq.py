import numpy as np
import pandas as pd
from DVSL_code import dtw2
def DVSL_dtw(trainset,templatenum,reg_lambda,old_virtseq,gamma, classnum, trainsetnum, err_limit):

    max_nIter=200
    dim = trainset.shape[1]
    downdim = classnum*templatenum

    old_virtseq = np.array(old_virtseq)

    ##############################################################################
    ##############################INTIALIZATION###################################
    ##############################################################################
    R_A = np.zeros((dim,dim))
    R_B = np.zeros((dim,downdim))
    N = sum(trainsetnum)
    seq_counter=0
    for c in range(0,classnum):
        for n in range(0, trainsetnum[c]):
            seqlen=1
            T_ini = np.divide(np.ones((seqlen, templatenum)), seqlen * templatenum)
            for i in range(0,seqlen):
                temp_first=trainset.iloc[seq_counter].to_numpy().reshape((trainset.iloc[seq_counter].shape[0],1))
                temp_second=trainset.iloc[seq_counter].to_numpy().reshape((1,trainset.iloc[seq_counter].shape[0]))
                temp_ra=temp_first*temp_second
                virtseq_counter = 0;
                for j in range(0,templatenum):
                    R_A = R_A + T_ini[i][j]*temp_ra
                    R_B = R_B + T_ini[i][j]*temp_first*old_virtseq[c][j]
                    virtseq_counter = virtseq_counter + 1
            seq_counter=seq_counter+1


    # print(R_B)
    # print('***********************')
    # print(R_A)
##############################################################################
##################CHECK R_A VALUES WITH ACTUAL DATASET########################
##############################################################################

# CLOSEDFORM VAIRABLES, A^(-1) IN THE TEXT
    R_I = R_A + (reg_lambda*N*(np.identity(dim)))
    L_old =  np.linalg.lstsq(R_I, R_B,rcond=N)[0]

#     #############IF THE RESULTS ARE NOT GOOD LOOK INTO RCOND VALUE IN np.linalg.lstsq########

    #CLOSEDFORM VARIABLES, C IN THE TEXT
    B2=0

    for c1 in range(0,classnum):
        for n in range(0,trainsetnum[c1]):
            for c in range(0,classnum):
                if c!=c1:
                    for y in range(0, trainsetnum[c]):
                        for i in range(0,templatenum):
                            B2 = B2 +1;

    B2 = (sum(trainsetnum))*B2

    #####################UPDATE######################

    loss_old=10**8

    u=0
    new_virtseq = np.zeros(old_virtseq.shape)
    l=[]
    for nIter in range(0,max_nIter):
        loss=0
        R_A=np.zeros((dim,dim))
        R_B=np.zeros((dim,downdim))
        N=sum(trainsetnum)
        seq_counter = 0
        for c in range(0,classnum):
            for n in range(0,trainsetnum[c]):
                seqlen=trainset.iloc[seq_counter].to_numpy().reshape((1,trainset.iloc[seq_counter].shape[0])).shape[0]
                testvect_first=trainset.iloc[seq_counter].to_numpy().reshape((1,trainset.iloc[seq_counter].shape[0]))
                test_vector = np.matmul(testvect_first, L_old)
                test_vector=np.transpose(test_vector)
                dist, T = dtw2.dtw_fast(test_vector,np.transpose(old_virtseq[c]),False)
                loss = loss + dist
                for i in range(0, seqlen):
                    temp_first = trainset.iloc[seq_counter].to_numpy().reshape((trainset.iloc[seq_counter].shape[0], 1))
                    temp_second = trainset.iloc[seq_counter].to_numpy().reshape((1, trainset.iloc[seq_counter].shape[0]))
                    temp_ra = temp_first * temp_second
                    for j in range(0, templatenum):
                        R_A = R_A + T[i][j] * temp_ra
                        R_B = R_B + T[i][j] * temp_first * old_virtseq[c][j]
                seq_counter = seq_counter+1


        loss = loss/N + np.trace(np.matmul(np.transpose(L_old),L_old))

        print('Iteration ',nIter+1,' loss',loss)
        print('difference: ',abs(loss - loss_old))

        l.append(loss)

        if abs(loss - loss_old) < err_limit:
            break
        else:
            loss_old = loss


        R_I = R_A + (reg_lambda * N * (np.identity(dim)))

        L_temp = np.linalg.lstsq(R_I, R_B,rcond=N)[0]
        L_new = L_old - (gamma*(L_temp))
        L_old = L_new

        # new_virtseq = []
        temp_virt = Optimize_V_dtw(np.copy(old_virtseq),classnum,trainsetnum,L_new,trainset,templatenum,B2)

        for c in range(0,classnum):
            new_virtseq[c] = old_virtseq[c] - (gamma * temp_virt[c])

        old_virtseq = new_virtseq

    return L_new
def Optimize_V_dtw(virtual_sequence, classnum, trainsetnum,L,trainset,templatenum,B2):
    B1=0
    vjb=0
    vja=0

    N=sum(trainsetnum)

    for clnum in range(0, classnum):
        current_sequence = virtual_sequence[clnum]
        for j in range(0, templatenum):
            B1=0
            vjb=0
            vja=0
            seq_counter = 0
            for c in range(0,classnum):
                for n in range(0, trainsetnum[c]):
                    seqlen = trainset.iloc[seq_counter].to_numpy().reshape((1, trainset.iloc[seq_counter].shape[0])).shape[0]
                    testvect_first = trainset.iloc[seq_counter].to_numpy().reshape((1, trainset.iloc[seq_counter].shape[0]))
                    test_vector = np.matmul(testvect_first, L)
                    test_vector = np.transpose(test_vector)
                    dist, T = dtw2.dtw_fast(test_vector, np.transpose(current_sequence),False)

                    for i in range(0,seqlen):
                        B1 = B1 + T[i][j]
                        vja = vja + (T[i][j] * np.matmul(testvect_first, L))
                    seq_counter = seq_counter+1
            for cl in range(0, classnum):
                for n in range(0, trainsetnum[cl]):
                    for c in range(0, classnum):
                        if c!=n:
                            for y in range(0,trainsetnum[c]):
                                for i in range(0, templatenum):
                                    vjb = vjb + virtual_sequence[c][i]

            virtual_sequence[clnum][j] = (vja - (N*vjb))/(B1 - B2)


    return virtual_sequence
