
import nltk

import sys
reload(sys) 
sys.setdefaultencoding('UTF8')


prec=[]
for i in range(len(site)):
    print(i,len(site))
    prec.append([])
    for ll in range(len(abst)):
        print ll 
        words2=[]
        if(len(abst[ll])!=0):
            abst[ll]=str(re.sub(r'[^\w]',' ',abst[ll])) 
            words1=nltk.word_tokenize(abst[ll])
            x=nltk.pos_tag(words1)
            xx=[w[0] for w in x if w[1].startswith('N')]
            for word in xx:
               words2.append(text[i].count(word))
            if(max(words2)!=0):
               prec[i].append((str(l[ll]),max(words2)))
    prec[i].sort(key=lambda tup: tup[1])
    if(len(prec[i])>50):
        prec[i]=prec[i][:50]           

d=open("/home/raka/stats/rank1.txt",'w')
for i in range(len(prec)):
    d.write('\n'*2+"***"+"SITE:-"+str(i)+'\n'*2)
    for j in range(len(prec[i])):
        d.write("AA"+prec[i][j])
        d.write('\n') 

'''
precision=[]
for i in range(len(prec)):
'''
    
