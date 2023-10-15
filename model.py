from layer import Neuron, one_hot_encoding
from matrix_control import dimMatrix3to2, matrixDotProduct, multiplyScalar, printMatrix, reshapeMatrix, sumMatrix, transposeMatrix, splitMatrix, subtractMatrix, element_wise_multiply
from utils import ReLU, ReLU_derivatived

class LinearModel:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, sample_num, learning_rate):
        print("모델 생성 시작. 입력차원 {}개, 1히든차원 {}개, 2히든차원 {}개, 출력층 {}개로, 샘플 개수는 {}개 입니다."
              .format(input_size, hidden_size1, hidden_size2, output_size, sample_num));
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.sample_num = sample_num
        self.learning_rate = learning_rate

        self.layer_1 = [
            Neuron(input_size=self.input_size, output_size=self.hidden_size1, sample_num=self.sample_num)
            for _ in range(self.hidden_size1)
        ]
        self.hidden_output_1 = []

        self.layer_2 = [
            Neuron(input_size=self.hidden_size1, output_size=self.hidden_size2, sample_num=self.sample_num)
            for _ in range(self.hidden_size2)
        ]
        self.hidden_output_2 = []

        self.layer_3 = [
            Neuron(input_size=self.hidden_size2, output_size=self.output_size, sample_num=self.sample_num)
            for _ in range(self.output_size)
        ]
        self.final_output = []

    def preprocessing(self, X):
        for i in range(self.sample_num):
            X[i] = reshapeMatrix(X[i], 256, 1);
        return X

    def forward(self, X, label):
        # 순전파
        self.propagate(X)

        # 역전파
        self.backpropagate(label=label)


    # 순전파 과정
    def propagate(self, X):
        
        # 데이터 전처리
        X = self.preprocessing(X)

        # 첫 입력
        # 입력 사이즈 = (10 X 16 X 16) -> (10 X 256) -> Transpose. (256, 10)
        # 1번 가중치 = (96, 256)
        # 첫 순전파 = 1번 가중치 X 입력 = (96, 256) X (256, 10) = (96, 10)
        # 뉴런 별 = (1, 10) 96개

        # 2번 가중치 = (48, 96)
        # 첫 순전파 = 2번 가중치 X 입력 = (48, 96) X (96, 10) = (48, 10)
        # 뉴런 별 = (1, 10) 48개

        # 3번 가중치 = (7, 48)
        # 첫 순전파 = 3번 가중치 X 입력 = (7, 48) X (48, 10) = (7, 10)
        # 뉴런 별 = (1, 10) 7개
        
        for i in range(self.hidden_size1):
            target_neuron = self.layer_1[i]
            temp_output = []
            for j in range(self.sample_num):
                # 순전파
                Z = target_neuron.forward(X[j])
                # ReLU 함수
                A = max(0, Z[0][0])
                
                # A의 사이즈는 (1X1)이다. 현재 반복문이 끝나면 현 뉴런의 총 샘플 별 활성화 값이 저장됨.
                temp_output.append(A)
                # temp_output의 사이즈는 (1Xsample_num)
                
            self.hidden_output_1.append(temp_output)
        self.hidden_output_1 = splitMatrix(self.hidden_output_1)
        print("output size : {} X {}".format(len(self.hidden_output_1[0]), len(self.hidden_output_1)))
        # print('===============================');


        for i in range(self.hidden_size2):
            target_neuron = self.layer_2[i]
            temp_output = []
            for j in range(self.sample_num):
                # 순전파
                Z = target_neuron.forward(self.hidden_output_1[j])
                # ReLU 함수
                A = max(0, Z[0][0])

                temp_output.append(A)

            self.hidden_output_2.append(temp_output)
        self.hidden_output_2 = splitMatrix(self.hidden_output_2)
        print("output size : {} X {}".format(len(self.hidden_output_2[0]), len(self.hidden_output_2)))
        # print('===============================');


        for i in range(self.output_size):
            target_neuron = self.layer_3[i]
            temp_output = []
            for j in range(self.sample_num):
                # 순전파
                Z = target_neuron.forward(self.hidden_output_2[j])
                # ReLU 함수
                A = max(0, Z[0][0])

                temp_output.append(A)

            self.final_output.append(temp_output)
        self.final_output = splitMatrix(self.final_output)
        print("output size : {} X {}".format(len(self.final_output[0]), len(self.final_output)))
        print(self.final_output[0])
        # print('===============================');
    
    def backpropagate(self, label):
        self.labeled_list = one_hot_encoding(label)
        self.chainRule()
        # dim_changed_matrix_1 = dimMatrix3to2(self.hidden_output_1)
        # dim_changed_matrix_2 = dimMatrix3to2(self.hidden_output_2)
        # dim_changed_matrix_3 = dimMatrix3to2(self.final_output)

        
        # dw_3 =  matrixDotProduct(matrix1=subtractMatrix(matrix1=dim_changed_matrix_3, matrix2=self.labeled_list), matrix2=transposeMatrix(dim_changed_matrix_2))
        # print(len(dw_3))
        # print(len(dw_3[0]))
        # print(dw_3[0])
        # print(self.layer_3[0].weight_matrix)
        # for i in range(self.output_size):
        #     self.layer_3[i].weight_matrix = subtractMatrix(matrix1=self.layer_3[i].weight_matrix, matrix2=multiplyScalar([dw_3[i]], self.learning_rate/self.sample_num))
        # print(self.layer_3[0].weight_matrix)


        # dw_2 =  matrixDotProduct(matrix1=subtractMatrix(matrix1=dim_changed_matrix_2, matrix2=self.labeled_list), matrix2=transposeMatrix(dim_changed_matrix_1))
        # print(len(dw_2))
        # print(len(dw_2[0]))
        # print(dw_2[0])
        # print(self.layer_2[0].weight_matrix)
        # for i in range(self.output_size):
        #     self.layer_2[i].weight_matrix = subtractMatrix(matrix1=self.layer_2[i].weight_matrix, matrix2=multiplyScalar([dw_3[i]], self.learning_rate/self.sample_num))
        # print(self.layer_2[0].weight_matrix)


        # dw_1 =  matrixDotProduct(matrix1=subtractMatrix(matrix1=dim_changed_matrix_1, matrix2=self.labeled_list), matrix2=transposeMatrix(dim_changed_matrix_1))
        # print(len(dw_1))
        # print(len(dw_1[0]))
        # print(dw_1[0])
        # print(self.layer_1[0].weight_matrix)
        # for i in range(self.output_size):
        #     self.layer_1[i].weight_matrix = subtractMatrix(matrix1=self.layer_1[i].weight_matrix, matrix2=multiplyScalar([dw_3[i]], self.learning_rate/self.sample_num))
        # print(self.layer_1[0].weight_matrix)

    def chainRule(self):
        dim_changed_matrix_1 = dimMatrix3to2(self.hidden_output_1)
        dim_changed_matrix_2 = dimMatrix3to2(self.hidden_output_2)
        dim_changed_matrix_3 = dimMatrix3to2(self.final_output)

        # dz3
        dz_3 = subtractMatrix(matrix1=dim_changed_matrix_3, matrix2=self.labeled_list)
        # dw3
        dw_3 =  matrixDotProduct(matrix1=dz_3, matrix2=transposeMatrix(dim_changed_matrix_2))
        print(self.layer_3[0].weight_matrix)

        # dz2
        dz_2 = []
        for i in range(self.hidden_size2):
            a = transposeMatrix(self.layer_2[i].weight_matrix)
            print(len(a))
            print(len(a[0]))
            print(len(dz_3))
            print(len(dz_3[0]))
            dz_2.append(
                element_wise_multiply(matrixDotProduct(transposeMatrix(self.layer_3[i].weight_matrix), dz_3), ReLU_derivatived(self.hidden_output_2)))
        print(len(dz_2))
        print(len(dz_2[0]))

        # dw2

        # dz1

        # dw1
