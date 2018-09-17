'''
from bs4 import BeautifulSoup
import urllib2
f=open("sites.txt")
z=f.read()
zz=z.split('\n'+'\n')
refname=[[],[],[],[],[],[],[],[]]
time=[[],[],[],[],[],[],[],[]]
refname2=[[],[],[],[],[],[],[],[]]
time2=[[],[],[],[],[],[],[],[]]
'''
for w in range(101,len(zz)):
    print(w)
    r=zz[w].split('\n')
    refname.append([])
    time.append([])
    refname2.append([])
    time2.append([])
    if(len(r)>0):
        j=len(r)-1  
        while (j>=0):
           print(j) 
           for s in range(len(refname[w])):
              print(refname[w][s]+"  " +time[w][s])      
           if(r[j].startswith('http')):
              response=urllib2.urlopen(r[j])
              links=response.read()
              soup=BeautifulSoup(links)
              lis=soup.find(id='pagehistory').find_all('li')
              for m in lis:
                   hists=m.find_all('a')
                   if(hists[1].text!='prev'):
                        continue
                   else:
                         response=urllib2.urlopen('http://en.wikipedia.org'+hists[1].get('href'))
                         ttime=hists[2].text
                         link=response.read()
                         soup2=BeautifulSoup(link)
                         ref=soup2.find(id='References')
                         if(ref!=None):
                              if(str(type(ref.next_element.next_element.next_element))=="<class 'bs4.element.Tag'>"):
                                     references=ref.next_element.next_element.next_element.find_all('li')
                                     for s in references:
                                             refname[w].append(s.text)
                                             time[w].append(ttime)
           j=j-1
    refname2[w]=set(refname[w])
    refname2[w]=list(refname2[w])
    if(len(refname2[w])>0):
        for s in range(len(refname2[w])):
           time2[w].append(time[w][refname[w].index(refname2[w][s])])             
