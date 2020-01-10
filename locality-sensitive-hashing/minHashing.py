import math
from random import randint

"""
Creates characteristics matrix from sets of shingles.
"""
def createCharacteristicMatrix(sets):
    uniqueShingles = set().union(*sets)
    M = [[0 for x in range(len(sets))] for y in range(len(uniqueShingles))]
    for (row, element) in enumerate(uniqueShingles):
        for (column, s) in enumerate(sets):
            if element in s:
                M[row][column] = 1
    return M

"""
Creates signature matrix
"""
def minHash(sets, n):
    # get the characteristc matrix
    M = createCharacteristicMatrix(sets)
    # get n hash functions
    functions = getHashFunctions(len(M),n)
    # initialize signature matrix with all values as inf
    sm = [[9999999 for x in range(len(M[0]))] for y in range(n)]
    for r in range(len(M)):
        hashes = []
        for function in functions:
            hashes.append((function[0]*r + function[1]) % function[2])
        for c in range(len(M[0])):
            if M[r][c] == 1:
                for h in range(len(hashes)):
                    if hashes[h] < sm[h][c]:
                        sm[h][c] = hashes[h]
    return sm

"""
Generate n hash functions in format h(x) = (ax + b) mod p.
P is prime larger then n.
A and b and random integers less then number of shingles.
"""
def getHashFunctions(nrOfRows,n):
    hashFunctions = []
    prime = getFirstPrime(nrOfRows)
    for _ in range(n):
        a = randint(0, nrOfRows)
        b = randint(0, nrOfRows)
        hashFunctions.append((a, b, prime))
    return hashFunctions

"""
Returns the first prime number larger then n
"""
def getFirstPrime(n):
    n+=1
    while not isPrime(n):
        n+=1
    return n

"""
Checks if n is a prime number. Stolen from StackOverflow.
"""
def isPrime(n) :

    if (n <= 1) :
        return False
    if (n <= 3) :
        return True
    if (n % 2 == 0 or n % 3 == 0) :
        return False

    i = 5
    while(i * i <= n) :
        if (n % i == 0 or n % (i + 2) == 0) :
            return False
        i = i + 6

    return True
