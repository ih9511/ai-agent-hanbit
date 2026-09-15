def fibonacci(n):
    sequence = []
    a, b = 1, 1
    while len(sequence) < n:
        sequence.append(a)
        a, b = b, a + b
    return sequence

if __name__ == "__main__":
    n = 100  # 예를 들어, 처음 10개의 항을 원한다고 가정합니다.
    print(fibonacci(n))
