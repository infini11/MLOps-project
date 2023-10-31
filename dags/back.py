import math

import re


# regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
# regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Z]+(\.[A-Z|a-z]{2,})+')
regex = re.compile(r'\w+[.-_]+\w*@\w+\.(com|fr)')

def isValid(email):
    if re.fullmatch(regex, email):
      print("Valid email")
    else:
      print("Invalid email")
    
def factoriel(n):
    fact = 1
    for i in range(1, n+1):
        print("", fact, "*", i, "")
        fact = fact*i
    return fact

def factor(n):
    if n == 0 : return 1
    else: 
        return n*factor(n-1)

def nombre_premier(n):
    for i in range(2,int(n**0.5)+1) :
        if (n%i) == 0 :
            return False
    return True

def palaindrom(input):
    txt = input[::-1]
    if input == txt: return True
    else: return False

def nombre_heureux(n):
    etat = False
    limite = False
    while limite == False:
        somme = 0
        for i in str(n):
            somme += int(i)**2
        n = str(somme)
        if n == '1':
            etat =  True
            break
        if int(n) < 10:
            limite = True
    return etat

def fibonaci(n):
    if n == 0 : return 0
    elif n == 1 : return 1

    return fibonaci(n-1)+ fibonaci(n-2)

def fib(n):
    if n == 0 : return 0
    elif n == 1 : return 1
    somme = 0
    while n>0:
        somme = somme + 2*n-3
        n=n-1
    return somme


def tail(path):
    with open(path) as fp:
        compt = 1
        line = fp.readline()
        while line and compt <=10:
            print(line)
            line = fp.readline()
            compt = compt + 1

def difference(string1, string2):
      # Split both strings into list items
    # string1 = string1.split()
    # string2 = string2.split()

    A = set(string1) # Store all string1 list items in set A
    B = set(string2) # Store all string2 list items in set B
    
    str_diff = A.symmetric_difference(B)

    
    isEmpty = (len(str_diff) == 0)
    if isEmpty:
        print("No Difference. Both Strings Are Same")
    else:
        print("The Difference Between Two Strings: ")
        print(str_diff)
    
    print('The programs runs successfully.')


import threading

def salut():
    print(f"Salut")

def toto():
    print(f"Toto")

# création de thread
t1 = threading.Thread(target=salut)
t2 = threading.Thread(target=toto)

    
if __name__ == "__main__":
    print(factor(3))
    print(nombre_premier(2))
    print(palaindrom("kayak"))
    print(nombre_heureux(7))
    print(fibonaci(7))
    print(fib(7))
    isValid('sndm1995.devoteam@sdf.com')
    tail('/Users/smboup/Documents/Infini/training/MLOps/MLOps-project/docker-compose.yaml')
    difference("string1", "string2")
    
    t1.start()
    t2.start()

    t1.join()
    t2.join()

#Expression reguliere metacaractere

#Metacaracters . ^ $ * + ? { } [ ] \ | ( )

#[ et ] sont utilisés pour spécifier une classe de caractères, qui forme un ensemble de caractères dont vous souhaitez trouver la correspondance.
#   - Par exemple, [abc] ou [a-c] correspond à n'importe quel caractère parmi a, b ou c, 
#   - Les métacaractères (à l’exception de \) ne sont pas actifs dans les classes
#       - Par exemple, [akm$] correspond à n'importe quel caractère parmi 'a', 'k', 'm' ou '$' ; '$' 
#          est habituellement un métacaractère mais dans une classe de caractères, il est dépourvu de sa signification spéciale.
#  
# '^' exception 
#   - Par exemple, [^5] correspond à tous les caractères, sauf '5'. Si le caret se trouve ailleurs dans la classe de caractères, 
#       il ne possède pas de signification spéciale. Ainsi, [5^] correspond au '5' ou au caractère '^'.
#
# \ annule la signification speciale
#   - Elle est aussi utilisée pour échapper tous les métacaractères afin d'en trouver les correspondances dans les motifs ; 
#       par exemple, si vous devez trouver une correspondance pour [ ou \, vous pouvez les précéder avec une barre oblique 
#       inverse pour annuler leur signification spéciale : \[ ou \\.
#
#  \d : Correspond à n'importe quel caractère numérique ; équivalent à la classe [0-9].
#  \D : Correspond à n'importe quel caractère non numérique ; équivalent à la classe [^0-9].
#  \s : Correspond à n'importe quel caractère « blanc » ; équivalent à la classe [ \t\n\r\f\v].
#  \S : Correspond à n'importe quel caractère autre que « blanc » ; équivalent à la classe [^ \t\n\r\f\v].
#  \w : Correspond à n'importe quel caractère alphanumérique ; équivalent à la classe [a-zA-Z0-9_].
#  \W : Correspond à n'importe quel caractère non-alphanumérique ; équivalent à la classe [^a-zA-Z0-9_].
#  Le dernier métacaractère de cette section est .. Il correspond à tous les caractères, à l'exception du caractère de 
# retour à la ligne ; il existe un mode alternatif (re.DOTALL) dans lequel il correspond également au caractère de retour 
# à la ligne. . est souvent utilisé lorsque l'on veut trouver une correspondance avec « n'importe quel caractère ».
# '*' signifie que le caractere precedent peut etre 0 ou plusieurs
# '+' requiert au moins une occurrence à la difference de '*' qui ne requiert pas une occurence
# '?' repetion de 0 ou une fois.
# Le plus compliqué des quantificateurs est {m,n} où m et n sont des entiers décimaux. 
#   Ce quantificateur indique qu'il faut au moins m répétitions et au plus n.
#   - Par exemple, a/{1,3}b fait correspondre 'a/b', 'a//b' et 'a///b'. 
#       Elle ne fait pas correspondre 'ab' (pas de barre oblique) ni 'a////b' (quatre barres obliques).
#  Recherche de correspondances : match(), search(), findall(), finditer()
#
#  group() : Renvoie la chaîne de caractères correspondant à la RE
#  start() : Renvoie la position de début de la correspondance
#  end() : Renvoie la position de fin de la correspondance
#  span() : Renvoie un n-uplet contenant les positions (début, fin) de la correspondance
# 
# match() : Détermine si la RE fait correspond dès le début de la chaîne.
# search() : Analyse la chaîne à la recherche d'une position où la RE correspond.
# findall() : Trouve toutes les sous-chaînes qui correspondent à la RE et les renvoie sous la forme d'une liste.
# finditer() : Trouve toutes les sous-chaînes qui correspondent à la RE et les renvoie sous la forme d'un itérateur.
# 
# ^
#   Correspond à un début de ligne. Par exemple, si vous voulez trouver le mot From uniquement quand il est en début de ligne, la RE à utiliser est ^From.
#   print(re.search('^From', 'From Here to Eternity'))  
#   <re.Match object; span=(0, 4), match='From'>
# $
#    Correspond à une fin de ligne, ce qui veut dire soit la fin de la chaîne, soit tout emplacement 
#       qui est suivi du caractère de nouvelle ligne.
#       print(re.search('}$', '{block}'))  
#           <re.Match object; span=(6, 7), match='}'>
#       print(re.search('}$', '{block} '))
#           None
#       print(re.search('}$', '{block}\n'))  
#           <re.Match object; span=(6, 7), match='}'>
# \A
#       Correspond au début de la chaîne de caractères, uniquement
# \Z
#       Correspond uniquement à la fin d'une chaîne de caractères
# \b
#    Limite de mot. C'est une assertion de largeur zéro qui correspond uniquement aux positions de début et de fin de mot.. 
#  p = re.compile(r'\bclass\b')
#       print(p.search('no class at all'))
#           <re.Match object; span=(3, 8), match='class'>
#       print(p.search('the declassified algorithm')
# \B
#   Encore une assertion de largeur zéro, qui est l'opposée de \b, 
#   c'est-à-dire qu'elle fait correspondre uniquement les emplacements qui ne sont pas à la limite d'un mot.