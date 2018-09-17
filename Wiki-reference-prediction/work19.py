import sys
reload(sys) 
sys.setdefaultencoding('UTF8')
from socket import error as SocketError
import errno
import urllib2
from bs4 import BeautifulSoup
from dateutil import parser
rr=0
q=[]
q.append(a)
while(True):
    try: 
          response=urllib2.urlopen(a)
          break
    except SocketError as e:
          if e.errno!=errno.ECONNRESET:
               print("ERROR")
    except urllib2.URLError as err: 
           print("ERROR")
           pass 
#response=urllib2.urlopen(a)
p=response.read()
soup=BeautifulSoup(p)
j=[] 
j=soup.find_all('a')
for w in j:
    z=w.get('class')
    if(z!=None):
          if(z[0]=='mw-nextlink' and rr!=1):
               rr=1 
               q.append('https://en.wikipedia.org'+w.get('href'))
if(rr==0):
    q.append('0')

joey=0
while True:
   joey=joey+1
   print("A"+str(joey)) 
   rr=0 
   if(q[len(q)-1]=='0'):
      break
   while(True):
      try: 
          response=urllib2.urlopen(q[len(q)-1])
          break
      except SocketError as e:
               if e.errno!=errno.ECONNRESET:
                    print("ERROR")
      except urllib2.URLError as err: 
               print("ERROR")
               pass 
   p=response.read()
   soup=BeautifulSoup(p)
   j=[] 
   j=soup.find_all('a')
   for w in j:
          z=w.get('class')
          if(z!=None):
	       if(z[0]=='mw-nextlink' and rr!=1):
	            rr=1
	            flag=1   
	            q.append('https://en.wikipedia.org'+w.get('href'))
   if(rr==0):
          q.append('0')  
          break



flag=0
flag2=0

for i in range(len(q)):
   if(q[i].startswith('http')):
          while(True):
                   try: 
                        response=urllib2.urlopen(q[i])
                        break
                   except SocketError as e:
                        if e.errno!= errno.ECONNRESET:
                            print("ERROR")
                        pass # Handle error here.
                   except urllib2.URLError as err: 
                        print("ERROR")
                        pass            
          links=response.read()
          soup=BeautifulSoup(links)
          lis=soup.find(id='pagehistory').find_all('li')
          for m in lis:
                   hists=m.find_all('a')
                   if(len(hists)==0):
                        continue
                   if(hists[1].text!='prev'):
                        continue
                   else:
                         ttime=parser.parse(hists[2].text)
                         print("B"+str(ttime))
                         if(ttime.year==2014 and flag==0):
                            flag=1
                            print("K")
                            #ff.write(a2+'\n'*2)
                            while(True):
                                try: 
                                    response2=urllib2.urlopen('http://en.wikipedia.org'+hists[1].get('href'))
                                    break
                                except SocketError as e:
                                    if e.errno!= errno.ECONNRESET:
                                       print("ERROR")
                                    pass # Handle error here.
                                except urllib2.URLError as err: 
                                    print("ERROR")
                                    pass 
                            response2=response2.read()
                            soup2=BeautifulSoup(response2)
                            soup3=soup2 
                            #ff.write("*************************************"+'\n'*6)
                            g2=soup2.find_all('p')
                            grefs=soup2.find_all('li')
                            #ff.write('\n'*2+"REFERENCES"+'\n'*2) 
                            counter1=0
                            refs2014=[]
                            for gg2 in grefs:
                                if(gg2.get('id')!=None):
                                    if(gg2.get('id').startswith('cite')):
                                       counter1=counter1+1 
                                       #ff.write(gg2.text+'\n')
                                       refs2014.append(str(gg2.text))
                            #ff.write("*************************************"+'\n'*6)    
                         if(ttime.year==2012 and flag2==0 and flag==1):
                            flag2=1 
                            n=n+1
                            gg.write(a2+'\n'*2)
                            while(True):
                                try: 
                                    response2=urllib2.urlopen('http://en.wikipedia.org'+hists[1].get('href'))
                                    break
                                except SocketError as e:
                                    if e.errno != errno.ECONNRESET:
                                       print("ERROR") 
                                    pass # Handle error here.
                                except urllib2.URLError as err: 
                                    print("ERROR")
                                    pass   
                            response2=response2.read()
                            soup2=BeautifulSoup(response2)
                            gg.write("*************************************"+'\n'*6)
                            g2=soup2.find_all('p')
                            for gg2 in g2:
                                gg.write("A "+gg2.text+'\n')
                            grefs=soup2.find_all('li')
                            #gg.write('\n'*2+"REFERENCES"+'\n'*2)
                            counter2=0
                            refs2012=[]  
                            for gg2 in grefs:
                                if(gg2.get('id')!=None):
                                    if(gg2.get('id').startswith('cite')):
                                       counter2=counter2+1
                                       #gg.write(gg2.text+'\n')
                                       refs2012.append(str(gg2.text))
                            execfile('work21.py')
                            gg.write("*************************************"+'\n'*6)    
                            break  
          if(flag2==1):
                   break               

'''
g=open("strange.txt",'w')
g.write(soup.get_text())
'''


