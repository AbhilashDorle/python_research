import numpy as np
import DVSL_code

virtual_sequence = []


class DVSL:

    def __init__(self, reg_lambda, gamma, classnum, templatenum, err):
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.classnum = classnum
        self.templatenum = templatenum
        self.err = err

    def generate_virtseq(self, reg_lambda, gamma, classnum, templatenum, err):
        downdim = classnum * templatenum
        for i in range(0, classnum):
            virtual_sequence.append(np.random.randn(templatenum, downdim))

        # vs = [[-1.0689,-2.9443,0.3252,1.3703],
        #       [-0.8095,1.4384,-0.7549,-1.7115]]
        # virtual_sequence.append(vs)
        #
        # vs = [[-0.1022,0.3192,-0.8649,-0.1649],
        #       [-0.2414,0.3129,-0.0301,0.6277]]
        # virtual_sequence.append(vs)
        # vs=[[0.3252,1.3703,-0.1022,0.3192,-0.8649,-0.1649],[-0.7549,-1.7115,-0.2414,0.3129,-0.0301,0.6277]]
        # virtual_sequence.append(vs)
        #
        # vs=[[1.0933,-0.8637,-1.2141,-0.0068,-0.7697,-0.2256],[1.1093,0.0774,-1.1135,1.5326,0.3714,1.1174]]
        # virtual_sequence.append(vs)

        # vs = [[-1.0891,0.5525,1.5442,-1.4916,-1.0616,-0.6156],[0.0326,1.1006,0.0859,-0.7423,2.3505,0.7481]]
        #
        # virtual_sequence.append(vs)


        # ts = DVSL_code.EVA_MATRIX.Mat(reg_lamda,templatenum,virtual_sequence,gamma,classnum,err)
        # ts.disp(reg_lamda, templatenum, virtual_sequence, gamma, classnum, err)
        #
        # print('Printing VS')
        for i in range(0, classnum):
            print(virtual_sequence[i])

        mat = DVSL_code.EVA_MATRIX.Mat(reg_lambda, templatenum, virtual_sequence, gamma, classnum, err)

        Acc = mat.evaluate_groundmetric(reg_lambda, templatenum, virtual_sequence, gamma, classnum, err)

        print('Accuracy is ',Acc)

lambda_list = [0.0007]
gamma_list = [0.006]
classes = 2
template_num = 2
err_rate = 0.01

obj_list = []
for lambdas in lambda_list:
    for gammas in gamma_list:
        obj_list.append(DVSL(lambdas, gammas, classes, template_num, err_rate))

for obj in obj_list:
    obj.generate_virtseq(lambdas, gammas, classes, template_num, err_rate)


