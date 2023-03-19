import numpy as np

diabetes = np.loadtxt('dane/diabetes.txt', dtype="str")
diabetes_type = np.loadtxt('dane/diabetes-type.txt', dtype="str")
info = np.loadtxt('dane/_info-data-discrete.txt', dtype="str")


# Zadanie 3a
klasy = np.unique(diabetes[:, -1])

print("3a. Symbole klas decyzyjnych:")
for c in klasy:
    print(c)


# Zadanie 3b

klasy = np.unique(diabetes[:, -1], return_counts=True)

print("3b. Wielkości klas decyzyjnych:")
x, y = np.where(info == "diabetes")
decision_class_size = int(info[int(x)][2])
print(f"wielkosc klas decyzyjnych = {decision_class_size}")


# Zadanie 3c
print("\n3c\n")
x1, y1 = np.where(diabetes_type == "n")
for i in x1:
    tym = np.array(diabetes[:, i], dtype='float')
    max1 = np.max(tym)
    min1 = np.min(tym)
    print(f"{diabetes_type[i][0]}: max = {max1}, min = {min1}")