def count(arr):
    consonents  = ['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z']
    vowels = ['a','e','i','o','u']
    c_count = 0
    v_count = 0
    user_input=arr.lower()
    for i in user_input:
        if i in vowels:
            v_count+=1
        if i in consonents:
            c_count+=1
    return v_count,c_count        


arr = input ("Enter the string")

v,c=count(arr)
print("Vowels : ",v)
print("Consonents : ",c)



