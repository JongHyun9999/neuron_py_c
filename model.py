from layer import Neuron, one_hot_encoding
from matrix_control import dimMatrix3to2, matrixDotProduct, multiplyScalar, printMatrix, reshapeMatrix, sumMatrix, transposeMatrix, splitMatrix, subtractMatrix, element_wise_multiply
from utils import ReLU, ReLU_derivatived, sigmoid, Softmax
import random

class LinearModel:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, sample_num, learning_rate, epochs):
        print("모델 생성 시작. 입력차원 {}개, 1히든차원 {}개, 2히든차원 {}개, 출력층 {}개로, 샘플 개수는 {}개 입니다."
              .format(input_size, hidden_size1, hidden_size2, output_size, sample_num));
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.sample_num = sample_num
        self.learning_rate = learning_rate
        self.epochs = epochs

        start = -1
        end = 1

        self.w1 = [[random.uniform(start, end) for _ in range(input_size)] for _ in range(hidden_size1)];
        self.w2 = [[random.uniform(start, end) for _ in range(hidden_size1)] for _ in range(hidden_size2)];
        self.w3 = [[random.uniform(start, end) for _ in range(hidden_size2)] for _ in range(output_size)];

    def preprocessing(self, X):
        for i in range(len(X)):
            for j in range(len(X[0])):
                X[i][j] = X[i][j] / 255
        
        return X


    def forward(self, X, label):
        X = self.preprocessing(X)
        for i in range(self.epochs): 
            print('epochs : {}'.format(i+1))
            dict = self.propagate(X)
            self.backpropagate(label=label, dict=dict)
        
        for i in range(self.output_size):
            print(self.w3[i][0])


    def propagate(self, X):
        dict = {}

        z1 = matrixDotProduct(self.w1, X)
        a1 = ReLU(z1)

        z2 = matrixDotProduct(self.w2, a1)
        a2 = ReLU(z2)

        z3 = matrixDotProduct(self.w3, a2)
        a3 = sigmoid(z3)

        scaled_a3 = Softmax(transposeMatrix(a3))

        dict = {
            'a0' : X,
            'z1' : z1,
            'a1' : a1,
            'z2' : z2,
            'a2' : a2,
            'z3' : z3,
            'a3' : scaled_a3
        }

        return dict

    
    def backpropagate(self, label, dict):
        labeled_list = one_hot_encoding(label=label)

        # dz, dw 구하기. chain rule
        dz_3 = transposeMatrix( subtractMatrix( matrix1=transposeMatrix(dict['a3']), matrix2=labeled_list ) )
        dw_3 = matrixDotProduct(dz_3, transposeMatrix(dict['a2']))
        da_2 = matrixDotProduct(transposeMatrix(self.w3), dz_3)

        dz_2 = element_wise_multiply( da_2, ReLU_derivatived( dict['z2'] ) )
        dw_2 = multiplyScalar( matrixDotProduct ( dz_2, transposeMatrix(dict['a1']) ), 1/self.sample_num)
        da_1 = matrixDotProduct( transposeMatrix(self.w2), dz_2 )

        dz_1 = element_wise_multiply( da_1, ReLU_derivatived( dict['z1'] ) )
        dw_1 = multiplyScalar( matrixDotProduct ( dz_1, transposeMatrix(dict['a0']) ), 1/self.sample_num)
        da_1 = matrixDotProduct( transposeMatrix(self.w1), dz_1 )

        # 가중치 업데이트
        self.w1 = subtractMatrix(self.w1, multiplyScalar(matrix=dw_1, scalar=self.learning_rate))
        self.w2 = subtractMatrix(self.w2, multiplyScalar(matrix=dw_2, scalar=self.learning_rate))
        self.w3 = subtractMatrix(self.w3, multiplyScalar(matrix=dw_3, scalar=self.learning_rate))
    
    def test(self, test_data, test_label):
        # 테스트 데이터를 받아서 검증 작업을 실시한다.
        dict = self.propagate(X=test_data)

        return dict['a3']