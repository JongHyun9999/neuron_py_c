import random
from matrix_control import transposeMatrix, reshapeMatrix, matrixDotProduct, printMatrix

class Neuron:
    # 각 뉴런 별로 
    def __init__(self, input_size, output_size, sample_num):
        self.input_size = input_size
        self.output_size = output_size
        self.sample_num = sample_num
        self.weight_matrix = [[random.uniform(0, 0.1) for _ in range(input_size)]];
        # (1X256)

    def forward(self, x):
        # x 사이즈는 input_size X sample_num
        # tranposed_x = transposeMatrix(x)
        # print("가중치 : {} X {} ".format(len(tranposed_matrix), len(tranposed_matrix[0])))
        # print("입력 데이터 : {} X {} ".format(len(x), len(x[0])))
        self.output_matrix = matrixDotProduct(self.weight_matrix, x)
        return self.output_matrix




def one_hot_encoding(label):
    label_t = [1, 0, 0, 0, 0, 0, 0]
    label_u = [0, 1, 0, 0, 0, 0, 0]
    label_v = [0, 0, 1, 0, 0, 0, 0]
    label_w = [0, 0, 0, 1, 0, 0, 0]
    label_x = [0, 0, 0, 0, 1, 0, 0]
    label_y = [0, 0, 0, 0, 0, 1, 0]
    label_z = [0, 0, 0, 0, 0, 0, 1]

    labeled_list = []
    for i in range(len(label)):
        target_label = label_t
        if(label[i] == 'u'):
            target_label = label_u
        elif(label[i] == 'v'):
            target_label = label_v
        elif(label[i] == 'w'):
            target_label = label_w
        elif(label[i] == 'x'):
            target_label = label_x
        elif(label[i] == 'y'):
            target_label = label_y
        elif(label[i] == 'z'):
            target_label = label_z
        
        labeled_list.append(target_label)
    
    return transposeMatrix(labeled_list)


# class ReLU:
#     def __init__(self):
#         self.mask = None
    
#     def forward(self, x):
#         self.mask = (x<=0)
#         out = x.copy()
#         out[self.mask] = 0
    
#     def backward(self, dout):
#         dout[self.mask] = 0
#         dx = dout

#         return dx

