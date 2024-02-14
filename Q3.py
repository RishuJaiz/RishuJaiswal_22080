def CommonElementsInLists(a,b):
    count=0
    for i in a:
        for j in b:
            if i==j:
                count+=1
    return count

list1=list()
list2=list()
num1=int(input("Enter no of elements in list 1 : "))
num2=int(input("Enter no of elements in list 2 : "))
for i in range(num1):
    inputval=int(input("Enter Numbers For list 1"))
    list1.append(n1)
for i in range(num2):
    inputval=int(input("Enter Numbers For List 2"))
    list2.append(n2)
result=CommonElementsInLists(list1,list2)
print("Common Elements among Lists Count : ",result)
