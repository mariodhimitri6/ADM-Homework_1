# INTRODUCTION - Say "Hello, World!" With Python
print("Hello, World!")

#INTRODUCTION - Loops
if __name__ == '__main__':
    n = int(input())
    
    i=0
    while (i<n):   
        print(i**2)
        i=i+1

#INTRODUCTION - Python Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    div_1=a//b
    div_2=a/b
    print(div_1)
    print(div_2)
    
#INTRODUCTION - Arithmetic operations
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    sum=a+b
    difference=a-b
    product=a*b
    print(sum)
    print(difference)
    print(product)

#INTRODUCTION - Write a function
def is_leap(year):
    leap = False
    
    if year%400==0:
        leap=True
    elif year%100==0:
        leap=False
    elif year%4==0:
        leap=True

    return leap

#INTRODUCTION - Python IF-ELSE
 import math
import os
import random
import re
import sys
if __name__ == '__main__':
    n = int(input().strip())
    if (n%2==1):
        print("Weird")
    else:
       if (n%2==0 and n>=2 and n<=5):
        print("Not Weird")
       elif (n%2==0 and n>=6 and n<=20):
           print("Weird")
       elif (n%2==0 and n>20):
            print("Not Weird")
        


#INTRODUCTION - Print a Function
if __name__ == '__main__':
    n = int(input())
    i=1
    while (i<n+1):
        print(i,end="")
        i+=1


#STRINGS - sWAP cASE
def swap_case(s):
    return s.swapcase() 

#STRINGS - Whats your name?
 def print_full_name(a, b):
    print("Hello "+a +" "+b+"!"+" You just delved into python.")

#STRINGS - String Split and Join
def split_and_join(line):
    line_1=line.split(" ")
    line_2="-".join(line_1)
    return line_2

#STRINGS - Capitalize
def solve(s):
    
    r=s.split()
    for i in r:
        s=s.replace(i,i.capitalize())
    return s
#STRINGS - String Validator
if __name__ == '__main__':
 input_data=input()
isalnum = False
isalpha = False
isdigit = False
islower = False
isupper = False
for i in input_data:
    if(i.isalnum()):
        isalnum=True
    if(i.isalpha()):
        isalpha=True
    if(i.isdigit()):
        isdigit=True
    if(i.islower()):
        islower=True
    if(i.isupper()):
        isupper=True
    
print(isalnum)
print(isalpha)
print(isdigit)
print(islower)
print(isupper)

   
#STRINGS - Find a String
def count_substring(string, sub_string):
    count=0
    for i in range (0,len(string)-len(sub_string)+1):
        if(string[i:len(sub_string)+i]==sub_string):
            count+=1

    return count

#STRINGS - Text Wrap
def wrap(string, max_width):
    return textwrap.fill(string,max_width)

 #STRINGS - Text Alignment
 #Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#STRINGS - The Minions Game
def minion_game(string):
    s = string
    kevin = 0
    stuart = 0
    for i in range(len(s)): #Iterating through the string, check if any of the words has vowels, then the scored gets increased for Kevin, otherwise it gets increased for Stuart. In the end compare the results and declare the winner.
        if s[i] in 'AEIOU':
            kevin = kevin+(len(s)-i)
        else:
            stuart = stuart+ (len(s)-i)

    if (kevin > stuart):
        print ("Kevin", kevin)
    elif kevin < stuart:
        print ("Stuart", stuart)
    else:
        print ("Draw")



#BASIC DATA TYPES - Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    tuple_data=tuple(integer_list)
    print(hash(tuple_data))

 #BASIC DATA TYPES - Find the percentage
 if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    for name,scores in student_marks.items():
        if name==query_name:
            print("{0:.2f}".format(*[sum(scores)/len(scores)]))

#BASIC DATA TYPES - Find the runner up
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    list_1 = set(arr) 
    list_1.remove(max(list_1)) 
    print(max(list_1))
#BASIC DATA TYPES - Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    student_scores = student_marks[query_name]
    sum_scores=sum(student_scores)
    len_scores=len(student_scores)
    avg=sum_scores/len_scores
    print("{0:.2f}".format(avg))

 #BASIC DATA TYPES - Lists
  if __name__=='__main__':
    L=[]
    n=int(input())
    for i in range(0,n):
        c=input().strip().split(' ')
        if(c[0]=="print"):
            print(L)
        elif(c[0]=="sort"):
            L.sort()
        elif(c[0]=="reverse"):
            L.reverse()
        elif(c[0]=="pop"):
            L.pop()
        elif(c[0]=="count"):
            L.count(int(c[1]))
        elif(c[0]=="index"):
            L.index(int(c[1]))
        elif(c[0]=="remove"):
            L.remove((int(c[1])))  
        elif(c[0]=="append"):
            L.append((int(c[1])))          
        elif(c[0]=="insert"):
            L.insert((int(c[1])),(int(c[2])))

#STRINGS - Merge the Tools
def merge_the_tools(string, k):
    list_1 = []
    n = len(string)
    slicing = n//k
    for i in range(0, n, k):
        s = ("")
        for j in string[i : i + k]:
            if j not in s:
                s += j
            else:
                 continue         
        print(s)



#BASIC DATA TYPES - Find the percentage
  if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    student_scores = student_marks[query_name]
    sum_scores=sum(student_scores)
    len_scores=len(student_scores)
    avg=sum_scores/len_scores
    print("{0:.2f}".format(avg))

#BASIC DATA TYPES - List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())


    print([[i,j,k] for i in range (x+1) for j in range(y+1) for k in range (z+1) if (i+j       +k!=n)])
                   

#itertools.permutations()
from itertools import permutations

a, b = input().split()
p=permutations(sorted(a),int(b))
for i in p:
    print(''.join(i))

#SETS - Introduction to Sets
def average(array):
     
    total=sum(set(array))
    avg=total/len(set(array))
    return avg

#Polar Coordinates
 import cmath as cm

z=complex(input())
print (abs(z))
print(cm.phase(z))


#SETS - Mutations
def mutate_string(string, position, character):
    data=list(string)
    for i in range(len(data)):
        if (i==position):
            data[position]=''.join(character)
    data_2=''.join(data)

    return data_2

#SETS - Set Mutation
n = int(input())
set_A = set(map(int,input().split()))
N = int(input())
for i in range(N):
    command = input().split()
    set_B = set(map(int,input().split()))
    if (command[0] == 'update'):
        set_A |= set_B
    elif (command[0] == 'intersection_update'):
        set_A &= set_B
    elif (command[0] == 'difference_update'):
        set_A -= set_B
    elif (command[0] == 'symmetric_difference_update'):
        set_A ^= set_B
print(sum(set_A))

#SETS - Check Strict Subset
A  = set(input().split())
n = int(input())
for i in range(n):
    check = True
    s = set(input().split())
    if (s&A != s) or (s == A):
        check = False
        break
print(check)

#SETS - No Idea
if __name__=='__main__':
    
    n, m = (int(i) for i in input().split())
    l = map(int, input().strip().split(' '))
    set_A = set(map(int, input().strip().split(' ')))
    set_B = set(map(int, input().strip().split(' ')))
    happiness = 0
    for i in l:              #Here we check if an input is part of the Set A or Set B. Based on the outcome, we increase or decrease the value of the happiness.
        if i in set_A:
            happiness += 1
        if i in set_B:
            happiness += -1
    print(happiness)

#SETS - Check Subset
for i in range(int(input())):
    a = int(input()); 
    set_A = set(input().split()) 
    b = int(input()); 
    set_B = set(input().split())
    subset=((set_A & set_B) == set_A)
    print (subset)

#SETS - No Idea
if __name__=='__main__':
    
    n, m = (int(i) for i in input().split())
    l = map(int, input().strip().split(' '))
    set_A = set(map(int, input().strip().split(' ')))
    set_B = set(map(int, input().strip().split(' ')))
    happiness = 0
    for i in l:              #Here we check if an input is part of the Set A or Set B. Based on the outcome, we increase or decrease the value of the happiness.
        if i in set_A:
            happiness += 1
        if i in set_B:
            happiness += -1
    print(happiness)

#SETS - The Capitans Room
n = int(input())
room_number_list = list(map(int, input().split()))
cr = (sum(set(room_number_list)) * n - sum(room_number_list)) // (n - 1)
print(cr)


#Numpy - Arrays
def arrays(arr):
    
    reversing=numpy.array(numpy.flip(arr),float)

    return reversing 

#NUMPY - Shape and reshape
 import numpy

np_array = numpy.array(input().split(),int)
shaped=numpy.reshape(np_array,(3,3))
print(shaped)

#NUMPY - Inner and Outer
import numpy

A = numpy.array(input().split(),int)
B = numpy.array(input().split(),int)

Inner=numpy.inner(A,B)
Outer=numpy.outer(A,B)

print(Inner)
print(Outer)

#NUMPY - Transpose and Flatten
import numpy

n,m=(map(int,input().split()))
array=[]

for i in range (n):
    array.append([int(j) for j in input().split()])

array=numpy.array(array)
transposed=numpy.transpose(array)
flatten=array.flatten()
print(transposed)
print(flatten)

#NUMPY - Min and Max
import numpy

n,m=map(int,input().split())
array=[]
for i in range(n):
    array.append([int(j) for j in input().split()])
array=numpy.array(array)
array=numpy.min(array, axis = 1)
print(numpy.max(array, axis = None))

#NUMPY - Concatenate
import numpy

N,M,P= map(int, input().split())
array_1=[]
array_2=[]
for i in range(N):
    array_1.append([int(j) for j in input().split()])
for i in range(M):
    array_2.append([int(j) for j in input().split()])


concatenation=numpy.concatenate((array_1,array_2),axis =0)
print(concatenation)

#NUMPY - Eye and Identity
import numpy
n,m = list(map(int, input().split()))
numpy.set_printoptions(sign=" ") #For this part I used the discussions section to get some help on seperating the values while priting
matrix = numpy.eye(n, m)
print(matrix)

#NUMPY - DOT AND CROSS
import numpy

a=int(input())
array1=[]
array2=[]
for i in range(a):
    array1.append([int(j) for j in input().split()])
for i in range(a):
    array2.append([int(j) for j in input().split()])

print(numpy.dot(array1,array2))


#NUMPY - Array Mathematics
import numpy

n,m = map(int,input().split())
array_1 = []
array_2=[]
for i in range(n):
    array_1.append([int(j) for j in input().split()])

for i in range(n):
    array_2.append([int(j) for j in input().split()])
np_array_1=numpy.array(array_1)
np_array_2=numpy.array(array_2)
print(np_array_1 + np_array_2) 
print(np_array_1 - np_array_2)
print(np_array_1 * np_array_2)
print(np_array_1 // np_array_2)
print(np_array_1 % np_array_2)
print(np_array_1 ** np_array_2)

#NUMPY - Zeros and Ones
import numpy
n = list(map(int, input().split()))
array_zero = numpy.zeros(n, dtype=numpy.int)
array_one = numpy.ones(n, dtype=numpy.int)
print(array_zero, array_one,sep="\n")

#NUMPY - Mean, Var and STD DEV
import numpy
n,m = map(int,input().split())
array_1 = []
for i in range(n):
    array_1.append([int(j) for j in input().split()])

np_array = numpy.array(array_1)
mean=numpy.mean(np_array,axis=1)
var=numpy.var(np_array,axis=0)
std=numpy.std(np_array,axis=None)

numpy.set_printoptions(legacy='1.13') #For this line of code I used the discussion

print(mean)
print(var)
print(std)

#NUMPY- Floor, Ceil and Rint
import numpy
input_1=map(float, input().split())
data=list(input_1)
array=numpy.array(data)
numpy.set_printoptions(sign=" ")
print(numpy.floor(array))
print(numpy.ceil(array))
print(numpy.rint(array))

#NUMPY - Linear Algebra
import numpy

n = int(input())
array=[]
array = numpy.array([input().split() for i in range(n)], dtype=float)
numpy.set_printoptions(legacy="1.13")
lin=numpy.linalg.det(array)
print(lin)

#NUMPY - Polynomials
import numpy
values=list(map(float,input().split()))
val = numpy.polyval(values,float(input()))
print(val)




#NUMPY - Sum and Prod
import numpy
N, M = map(int, input().split())
array = []
for i in range(N):
    array.append([int(j) for j in input().split()])

np_array=numpy.array(array)
sum_array=numpy.sum(np_array,axis=0)
product_sum=numpy.product(sum_array,axis=0)
print(product_sum)

#NUMPY - 


#SETS - Symmetric Difference
M=input()
set_A=set(map(int,input().split()))
N=input()
set_B=set(map(int, input().split()))
symmetric_difference=(set_A ^ set_B)
sorted_symmetric=sorted(symmetric_difference)
print (*sorted_symmetric,sep='\n')

#SETS - Set.add()
input_number=int(input())
set_data=set()
for i in range (input_number):
    set_data.add(input())
    
print (len(set_data))

#SETS - Set.union()
n = int(input())
s1 = set(map(int, input().split()))
m = int(input())
s2 = set(map(int, input().split()))
union_data=(s1 | s2)
print(len(union_data))

#SETS - Set.intersection()
n = int(input())
s1 = set(map(int, input().split()))
m = int(input())
s2 = set(map(int, input().split()))
intersection_data=(s1 & s2)
print(len(intersection_data))

#SETS - Set.difference()
n = int(input())
s1 = set(map(int, input().split()))
m = int(input())
s2 = set(map(int, input().split()))
difference_data=(s1 - s2)
print(len(difference_data))

#SETS - Set.discard(),remove(),pop
n = int(input())
s = set(map(int, input().split()))
command_number=int(input())
for i in range(command_number):
    a=input().split()
    if a[0]=='pop':
        s.pop()
    elif a[0]=='discard':
        s.discard(int(a[1]))
    elif a[0]=='remove':
        s.remove(int(a[1]))
total=sum(s)
print(total)

#SETS - Set.symmetric_difference()
n = int(input())
s1 = set(map(int, input().split()))
m = int(input())
s2 = set(map(int, input().split()))
symmetric_difference=(s1 ^ s2)
print(len(symmetric_difference))


#BUILT-INS Python Evaluation
from __future__ import print_function

eval(input())

#BUILT-INS Zipped

n, x = (int(x) for x in input().split())
studs = []
for i in range(x):
    studs.append([float(x) for x in input().split()])
    
for subj in zip(*studs):
    print(sum(subj)/x)

#BUILT-INS Any or All
n, array = (int(input()), input().split())
print(all(int(x)>0 for x in array) and any(x==x[::-1] for x in array)

#BUILT-INS Athlete Sort

import math
import os
import random
import re
import sys


if __name__ == '__main__':
    n, m = (int(i) for i in input().split())
array = []
for i in range(n):
    array.append([int(i) for i in input().split()])
k = int(input())

array.sort(key = lambda x: x[k])
for x in array:
    print(*x)

#BUILT-INS- ginortS

sorted_string=(sorted(input(), key=lambda c: (c.isdigit() - c.islower(), c in '02468', c)))

print(*sorted_string,sep='')


#DATE AND TIME - Calendar Module
import calendar

calendar.datetime
from datetime import date 
week_days=['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY']
input_data=list(map(int,input().split()))
data= date(input_data[2],input_data[0],input_data[1])
print(week_days[data.weekday()])


#PYTHON FUNCTIONALS - Reduce Function
def product(fracs):
    t = reduce(lambda numerator,denominator:numerator*denominator,fracs,1)
    return t.numerator, t.denominator

#PYTHON FUNCTIONALS - Map and Lambda Function
cube = lambda x: x**3

def fibonacci(n):
    l=[0,1]
    for i in range(2,n):
        l.append(l[i-1]+l[i-2]) #Fibonacci sequence is xn = xn-1 + xn-2 where xn is the number we want to check, xn-1 is the previous term and xn-2 is the term before that
    return (l[0:n])


#ERRORS AND EXCEPTIONS - Exceptions

n = int(input())
for i in range(0,n):
    try:
        a, b = map(int, input().split())
        print (a//b)
    except ZeroDivisionError as e:
        print ("Error Code:", e)
    except ValueError as e:
        print ("Error Code:", e)

#ERRORS AND EXCEPTIONS - Incorrect Regex
import re

n=int(input())
for i in range(n):
    try:
        print(bool(re.compile(input())))
    except re.error:
        print('False')


#COLLECTIONS - collections.Counter()
import collections as co
data = int(input())
quantity = co.Counter(map(int,input().split())) #Inicialize the Counter object
n = int(input())
count = 0
for i in range(n):
    (size, price) = map(int,input().split())
    if quantity[size] > 0: #In the for loop, when a customer buys a shoe, we decrease the quantity the store has, and increase the amount of money/profit.
        quantity[size] -= 1
        count += price
print (count)

#COLLECTIONS - DefaultDict Tutorial
from collections import defaultdict
n, m = map(int, input().split())
d = defaultdict(list)
for i in range(1, n + 1):
    d[input()].append(str(i)) #For this line I used the discussions section to get started with defaultdict
for i in range(m):
    print(' '.join(d[input()]) or -1)

#COLLECTIONS - Collections.deque()

import collections
n = int(input())
deque = collections.deque()
for i in range(n):
    c= list(input().strip().split())
    if c[0] == 'pop':
        deque.pop()
    elif c[0] == 'popleft':
        deque.popleft()
    elif c[0] == 'append':
        deque.append(int(c[1]))
    elif c[0] == 'appendleft':
        deque.appendleft(int(c[1]))
for i in deque:
    print(i,end=' ')

#COLLECTIONS - Word Order
from collections import Counter, OrderedDict


class OrderedCounter(Counter, OrderedDict):
    pass

array = []
n = int(input())
for i in range(n):
    array.append(input().strip())
count = OrderedCounter(array)
print(len(count))
for i in count:
    print(count[i],end=' ')

#COLLECTIONS -Collections.OrderedDict()

from collections import OrderedDict
n=int(input())
dictionary = OrderedDict()
for i in range(n):
    item, price = input().rsplit(' ', 1)
    dictionary[item] = dictionary.get(item, 0) + int(price)
[print(item, dictionary[item]) for item in dictionary]

#COLLECTIONS - Collections.namedtuple()

from collections import namedtuple
n = int(input())
fields = input()
student = namedtuple('stud',fields)
sum_grades = 0
for i in range(n):
    s = student(*input().split())
    sum_grades += float(s.MARKS)
mean=sum_grades/ n
print(mean)


# EXERCISE_2 ALGORITHMS - Kangaroo

import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    for n in range(10000):
        if((x1+v1)==(x2+v2)):
            return "YES"
        x1+=v1
        x2+=v2
    return "NO"
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#EXERCISE_2 ALGORITHMS Birthday Cake Candles
#!/bin/python3

import math
import os
import random
import re
import sys

def birthdayCakeCandles(ar):
        count=0
        maximum = max(ar)
        for i in range(len(ar)):
            if(ar[i]==maximum):
                count+=1
        return count


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(ar)

    fptr.write(str(result) + '\n')

    fptr.close()

#EXERCISE_2 VIRAL ADVERTISING

import math
import os
import random
import re
import sys

def viralAdvertising(n):
    start =5
    total=0
    for i in range(1,n+1):
        liked = start//2
        total+=liked
        start = liked*3
    return total


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#XML - Find the Score

def get_attr_number(node):
   return len(node.attrib) + sum(get_attr_number(child) for child in node) #For this exercise I used the help of the discussions since I had no previous experience over XML before.


#REGEX AND PARSING - Re.findall() & Re.finditer()
import re
s = input()
x = re.findall(r'(?<=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])([AEIOUaeiou]{2,})(?=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])',s) # I used Google to get this line of ascii characters
if x:
    print(*x, sep='\n')
else:
    print(-1)

#REGEX AND PARSING - Group(), Groups() & Groupdict()
import re
x = re.search(r'([a-zA-Z0-9])(?=\1)',input())
print(x.group(1) if x else -1)


#REGEX AND PARSING - Detecting Floating Point Number
from re import match, compile

pattern = compile('^[-+]?[0-9]*\.[0-9]+$')
for i in range(int(input())):
    check=bool(pattern.match(input()))
    print(check)

#REGEX AND PARSING - Regex Substitution

import re
n=int(input())
for i in range(n):
    s = ''
    s = re.sub(r'(?<= )&&(?= )','and',input())
    s = re.sub(r'(?<= )\|\|(?= )','or',s)
    print (s)

#REGEX AND PARSING - Re.split()

regex_pattern = r"[,.]"






