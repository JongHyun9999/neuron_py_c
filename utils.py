def ReLU(x):
    return max(0, x[0][0])

def ReLU_derivatived(matrix):
    # 결과를 저장할 빈 리스트 생성
    result = []

    # 행렬의 각 요소에 대해 ReLU 미분 연산 수행
    for row in matrix:
        result_row = [1 if element > 0 else 0 for element in row]
        result.append(result_row)

    return result