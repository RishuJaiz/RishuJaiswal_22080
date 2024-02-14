def countConsonentsAndVowels(arr):
    input=arr.lower()
    consonents_num = 0
    vowels_num = 0
    vowels = "aeiou"
    for char in user_input:
        if char in vowels:
            vowels_num+=1
    consonents_num=len(input)-vowels_num

v,c=count(arr)
print("Vowels : ",v)
print("Consonents : ",c)



