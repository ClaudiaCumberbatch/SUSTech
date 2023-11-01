length = input()
length = int(length)
arr = [2, 0, 2, 3]
for i in range(length):
    arr.append((arr[-2] * arr[-1])%10)
print(arr[-1])
