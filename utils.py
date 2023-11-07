from matrix_control import transposeMatrix
import math


def ReLU(matrix):
    # 결과를 저장할 빈 리스트 생성
    result = []

    for row in matrix:
        result_row = [max(0,element) for element in row]
        result.append(result_row)

    return result

def ReLU_derivatived(matrix):
    # 결과를 저장할 빈 리스트 생성
    result = []

    # 행렬의 각 요소에 대해 ReLU 미분 연산 수행
    for row in matrix:
        result_row = [1 if element > 0 else 0 for element in row]
        result.append(result_row)

    return result

def sigmoid(x):
    """
    시그모이드 함수를 안전하게 적용하는 함수
    :param x: 입력 값 (스칼라 또는 2차원 리스트)
    :return: 시그모이드 함수가 적용된 결과
    """
    if isinstance(x, (int, float)):
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1 + exp_x)
    elif isinstance(x, list):
        if isinstance(x[0], (int, float)):
            return [sigmoid(val) for val in x]
        elif isinstance(x[0], list):
            return [[sigmoid(val) for val in row] for row in x]
    else:
        raise ValueError("올바른 입력 형식이 아닙니다.")

def Softmax(matrix):
    softmaxed_matrix = []

    # (14X7)
    for i in range(len(matrix)):
        sum_value = 0
        result = []

        for j in range(len(matrix[0])):
            sum_value = sum_value + matrix[i][j]
        for j in range(len(matrix[0])):
            result.append(matrix[i][j]/sum_value)

        softmaxed_matrix.append(result)
    
    return transposeMatrix(softmaxed_matrix)

import math


def cross_entropy(prob_dist_true, prob_dist_predicted):
    """
    두 확률 분포 간의 크로스 엔트로피 계산
    :param prob_dist_true: 실제 확률 분포를 나타내는 리스트
    :param prob_dist_predicted: 예측 확률 분포를 나타내는 리스트
    :return: 크로스 엔트로피 값
    """
    if len(prob_dist_true) != len(prob_dist_predicted):
        raise ValueError("두 확률 분포는 같은 길이여야 합니다.")

    cross_entropy_val = 0
    for true_prob, predicted_prob in zip(prob_dist_true, prob_dist_predicted):
        if true_prob > 0:
            cross_entropy_val -= true_prob * math.log(predicted_prob, 2)

    return cross_entropy_val


def calAccuracy(result, label):
    index_list = []

    for rows in result:
        rows_max = rows[0]
        max_index = 0
        for i in range(1, len(rows)):
            if rows_max < rows[i]:
                rows_max = rows[i]
                max_index = i
        index_list.append(max_index)

    labeled_list = []
    for i in range(len(label)):
        target_label = 0
        if(label[i] == 'u'):
            target_label = 1
        elif(label[i] == 'v'):
            target_label = 2
        elif(label[i] == 'w'):
            target_label = 3
        elif(label[i] == 'x'):
            target_label = 4
        elif(label[i] == 'y'):
            target_label = 5
        elif(label[i] == 'z'):
            target_label = 6
        
        labeled_list.append(target_label)
    
    # for i in range(len(result)):
    #     print(f'result = {result[i]}')
    #     print(f'label = {label[i]}')
    #     print(f'resulted = {index_list[i]}')
    #     print(f'labeled = {labeled_list[i]}')

    

    # print(len(index_list))
    # print(len(labeled_list))
    count = 0
    for i in range(len(index_list)):
        if(index_list[i] == labeled_list[i]):
            count = count + 1
    
    print(f'정답 개수 : {count}')

    return count