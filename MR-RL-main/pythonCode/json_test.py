
import random

while True:
  x = random.randint(100, 200)
  y = random.randint(100, 200)
  z = random.randint(100, 200)
  m = 0.5
  n = 50
  #print(q)
  q1 =str(x)
  q2 = str(y)
  q3=str(z)
  q4 = str(m)
  q5= str(n)
  qq=q1+' '+q2+' '+q3+' '+q4+' '+q5+'\n'
  with open("statandstop.txt",mode='r',encoding='utf-8') as notes:
    a=notes.read()
  print(a)
 # qq=str[q]
 # with open("shujux.txt", mode='a',encoding='utf-8') as notex:
   # notex.write(qq)
   # notex.close()




