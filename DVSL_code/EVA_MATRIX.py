import numpy as np
import pandas as pd
from DVSL_code import Learn_Virtseq
from DVSL_code import NNClassifier_dtw

class Mat:
    def __init__(self, reg_lambda, templatenum, virtual_sequence, gamma, classnum, err):
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.classnum = classnum
        self.templatenum = templatenum
        self.err = err
        self.virtual_sequence = virtual_sequence

    def evaluate_groundmetric(self, reg_lambda, templatenum, virtual_sequence, gamma, classnum, err):
        traininglabel, trainingset, trainsetnum = self.train_data()

        testsetdata, testsetlabel, testsetnum=self.test_data()

        L = Learn_Virtseq.DVSL_dtw(trainingset,templatenum,reg_lambda, virtual_sequence,gamma, classnum, trainsetnum, err)

        Acc = self.classification_with_learned_metric(trainingset,testsetdata,classnum,trainsetnum,L,testsetlabel,testsetnum)

        return Acc

    def train_data(self):
        # print('train')
        train_data = pd.read_csv(r'/home//abhilash//Dr_Sheng_Li//Datasets//UCRArchive_2018//GunPoint//GunPoint_TRAIN.csv', header=None)
        train_data = train_data.rename(columns={0: 'label'})
        train_data = train_data.sort_values(by=['label'],kind='mergesort')
        traininglabel = train_data['label']

        trainingsett = train_data.iloc[:, 1:]

        trainingsett=trainingsett.reset_index(drop=True)

        count_dict = traininglabel.value_counts().to_dict()

        num_of_samples=[]
        for i in sorted(count_dict):
            num_of_samples.append(count_dict[i])



        return traininglabel, trainingsett, num_of_samples

    def test_data(self):
        test_data = pd.read_csv(r'/home//abhilash//Dr_Sheng_Li//Datasets//UCRArchive_2018//GunPoint//GunPoint_TEST.csv', header=None)
        test_data = test_data.rename(columns={0:'label'})
        # TAKE A LOOK AT THE NN ALGO AND CHECK IF ANYTHING HAS TO CHANGE HERE

        testsetdata = test_data.iloc[:, 1:]

        testsetlabel = test_data['label']

        testsetnum=test_data.shape[0]

        return testsetdata,testsetlabel,testsetnum

    def classification_with_learned_metric(self,trainset,testsetdata,classnum,trainsetnum,L,testsetlabel,testsetdatanum):

        traindownset = pd.DataFrame(index = np.arange(trainset.shape[0]), columns=np.arange(L.shape[1]))

        testdownset = pd.DataFrame(index = np.arange(testsetdata.shape[0]), columns=np.arange(L.shape[1]))

        count=0
        for i in range(0,trainset.shape[0]):
            count=count+1
            temp=trainset.iloc[i].to_numpy().reshape(1,trainset.shape[1])
            # print(np.matmul(temp,L))
            traindownset.iloc[i] = np.matmul(temp,L)
            # print(trainset.iloc[i])
        for i in range(0,testsetdata.shape[0]):
            temp=testsetdata.iloc[i].values.reshape(1, testsetdata.shape[1])
            testdownset.iloc[i] = np.matmul(temp,L)


        return NNClassifier_dtw.test(classnum, traindownset, trainsetnum, testdownset, testsetdatanum, testsetlabel)