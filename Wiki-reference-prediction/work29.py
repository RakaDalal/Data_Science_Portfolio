import sys
reload(sys) 
sys.setdefaultencoding('UTF8')
import nltk
import matplotlib.pyplot as plt
from collections import Counter

refs=[]
info=[]
oldrefset=[]
text=[]
abstract=[]

execfile('work25.py')

import math
from textblob import TextBlob as tb

def tf(word, blob):
    if(len(blob.split())!=0): 
       return (blob.split().count(word)*1.0/max(Counter(blob.split()).values()))
    else:
       return 0.0

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.split())

def idf(word, bloblist):
    return (math.log(len(bloblist)*1.0/(1 + n_containing(word, bloblist))))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)



mstat1=[]
bloblist=[]
for i in range(len(text)):
    for j in range(len(text[i])):
        text[i][j]=str(re.sub(r'[^\w]',' ',text[i][j]))
        bloblist.append(text[i][j])

bloblist=list(set(bloblist)) 
for r in refs:
    print(refs.index(r))  
    words2=[]
    r[0]=str(re.sub(r'[^\w]',' ',r[0]))
    words1=nltk.word_tokenize(r[0].lower())
    tid=0
    for word in words1:
        temp=0
        for i in range(len(text[refs.index(r)])): 
            temp=temp+tfidf(word,text[refs.index(r)][i],bloblist)
        temp=temp*1.0/len(text[refs.index(r)])
        tid=tid+temp
    print("A",tid)
    mstat1.append(tid*100.0/len(words1))


mstat2=[]
bloblist=[]
for i in range(len(oldrefset)):
    for j in range(len(oldrefset[i])):
        oldrefset[i][j]=str(re.sub(r'[^\w]',' ',oldrefset[i][j]))
        bloblist.append(oldrefset[i][j])

bloblist=list(set(bloblist)) 
for r in refs:
    print(refs.index(r))  
    words2=[]
    r[0]=str(re.sub(r'[^\w]',' ',r[0]))
    words1=nltk.word_tokenize(r[0].lower())
    tid=0
    for word in words1:
        temp=0
        for i in range(len(oldrefset[refs.index(r)])): 
            temp=temp+tfidf(word,oldrefset[refs.index(r)][i],bloblist)
        temp=temp/len(oldrefset[refs.index(r)])
        tid=tid+temp
    print("B",tid)
    mstat2.append(tid*100.0/len(words1))

mstat3=[]
bloblist=[]
for i in range(len(text)):
    for j in range(len(text[i])):
        text[i][j]=str(re.sub(r'[^\w]',' ',text[i][j]))
        bloblist.append(text[i][j])

bloblist=list(set(bloblist)) 
for r in refs:
    print(refs.index(r))  
    words2=[]
    if(len(abstract[refs.index(r)])!=0):
	    abst=str(re.sub(r'[^\w]',' ',abstract[refs.index(r)][0]))
	    words1=nltk.word_tokenize(abst.lower())
	    words1=nltk.word_tokenize(abst)
	    tid=0
	    for word in words1:
		temp=0
		for i in range(len(text[refs.index(r)])): 
		    temp=temp+tfidf(word,text[refs.index(r)][i],bloblist)
		temp=temp/len(text[refs.index(r)])
		tid=tid+temp
	    print("C",tid)
	    mstat3.append(tid*100.0/len(words1))



plt.hold(False)  
z=plt.hist(mstat1,bins=30,color='red')
plt.ylabel("Number of references")
plt.xlabel("Average tf-idf score of the words of the title corresponding to it's text (multiplied by 100)")
plt.ylim(0,max(z[0])+2)
plt.savefig("/home/raka/stats/mstat37.png")


plt.hold(False)  
z=plt.hist(mstat2,bins=30,color='red')
plt.ylabel("Number of references")
plt.xlabel("Average tf-idf score of the words of the title corresponding to it's old reference set (multiplied by 100)")
plt.ylim(0,max(z[0])+2)
plt.savefig("/home/raka/stats/mstat38.png")



plt.hold(False)  
z=plt.hist(mstat3,bins=30,color='red')
plt.ylabel("Number of references")
plt.xlabel("Average tf-idf score of the words of the abstract corresponding to it's text(multiplied by 100)")
plt.ylim(0,max(z[0])+2)
plt.savefig("/home/raka/stats/mstat39.png")






    
