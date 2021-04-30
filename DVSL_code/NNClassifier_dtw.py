import numpy as np
import pandas as pd
from DVSL_code import dtw2

def test(classnum,trainset,trainsetnum,testsetdata,testsetdatanum,testsetlabel):

    trainsetdatanum = sum(trainsetnum)
    trainsetlabel=[]
    for c in range(0,classnum):
        for per_sample_count in range(0, trainsetnum[c]):
            trainsetlabel.append(c+1)


    testsetlabelori = testsetlabel;
    # getLabel(testsetlabelori)

    k_pool = [1]

    k_num = len(k_pool)

    Acc = np.zeros((k_num,1))

    dis_totrain_scores = np.zeros((trainsetdatanum,testsetdatanum))

    ClassLabel = np.arange(1, classnum+1)

    dis_ap = np.zeros((1,testsetdatanum))

    right_num = np.zeros((k_num,1))

    counter =0

    for j in range(0,testsetdata.shape[0]):
        for m2 in range(0,trainset.shape[0]):

            t = trainset.iloc[m2].to_numpy().reshape((trainset.iloc[m2].shape[0],1))
            r = testsetdata.iloc[j].to_numpy().reshape((trainset.iloc[m2].shape[0],1))
            Dist= dtw2.dtw_fast(t, r, True)
            if np.isnan(Dist) == True:
                print('NaN distance')

            dis_totrain_scores[m2][j] = Dist

        index = np.argsort(dis_totrain_scores[:,j])
        distm = np.sort(dis_totrain_scores[:,j])

        # print(index)
        # print('***********************************')

        for k_count in range(0,k_num):
            cnt = np.zeros((classnum,1))
            for temp_i in range(0, k_pool[k_count]):
                for clabel in ClassLabel:
                    if clabel == trainsetlabel[index[temp_i]]:
                        ind = clabel
                cnt[ind-1] = cnt[ind-1] + 1
            ind = np.where(cnt==np.amax(cnt))[0][0]
            predict = ClassLabel[ind]
            # predict = predict[0]
            if predict == testsetlabelori[j]:
                counter = counter +1
                right_num[k_count] = right_num[k_count] + 1

        temp_dis = (-1)*dis_totrain_scores[:,j]
# #         # temp_dis[np.argwhere(np.isnan(temp_dis))]=0
# # #
        for dis in temp_dis:
            if np.isnan(dis):
                temp_dis[np.where(dis)]=0

    # print(right_num)
    Acc = np.divide(right_num,testsetdatanum)

    return Acc[0][0]
# # #
# # # # def getLabel(classid):
# # # #     X = np.zeros((classid.shape[0], max(classid)))
# # # #     for i in range(0, classid.shape[0]):
# # #         if(i == )