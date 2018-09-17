import sys
reload(sys) 
sys.setdefaultencoding('UTF8')
import nltk
import matplotlib.pyplot as plt

'''
refs=[]
info=[]
oldrefset=[]
text=[]
abstract=[]

execfile('work25.py')
'''

#mstat1
mstat1=[]
for r in refs:
    print(refs.index(r))  
    words2=[]
    words1=nltk.word_tokenize(r[0])
    x=nltk.pos_tag(words1)
    xx=[w[0] for w in x if w[1].startswith('V')]
    if(len(xx)!=0):
        for word in xx:
            for i in range(len(text[refs.index(r)])): 
                words2.append(text[refs.index(r)][i].count(word))
            print("A",max(words2))
            mstat1.append(max(words2))
    

#mstat2
mstat2=[]
for r in refs:
    print(refs.index(r))
    words2=[]
    words1=nltk.word_tokenize(r[0])
    x=nltk.pos_tag(words1)
    xx=[w[0] for w in x if w[1].startswith('V')]
    if(len(xx)!=0):
        for word in xx:
            for i in range(len(oldrefset[refs.index(r)])): 
                 words2.append(oldrefset[refs.index(r)].count(word))
        print("A",max(words2))
        mstat2.append(max(words2))


plt.hold(False)  
z=plt.hist(mstat1,bins=[0,5,10,15,20,25,30,35,40,45,50],color='red')
plt.ylabel("Number of references")
plt.xlabel("Max Common Verb of title with the text")
plt.ylim(0,max(z[0])+2)
plt.savefig("/home/raka/stats/mstat5.png")



plt.hold(False)  
z=plt.hist(mstat2,bins=8,color='red')
plt.xlabel("Number of references")
plt.ylabel("Max Common Verb with the old reference set")
plt.ylim(0,max(z[0])+2)
plt.savefig("/home/raka/stats/mstat6.png")



#mstat3
#mstat4
mstat3=[]
mstat4=[]

for r in refs[346:]:
    print(refs.index(r))
    words2=[]
    if(len(abstract[refs.index(r)])!=0):
       words1=nltk.word_tokenize(abstract[refs.index(r)][0])
       x=nltk.pos_tag(words1)
       xx=[w[0] for w in x if w[1].startswith('V')]
       k=0
       if(len(xx)!=0):
	       for word in xx:
		  flag=0 
		  for i in range(len(text[refs.index(r)])): 
		     words2.append(text[refs.index(r)][i].count(word))
		     if(flag==0 and text[refs.index(r)][i].count(word)!=0):
		           flag=1
		           k=k+1  
	       print(k,len(xx))
	       mstat3.append(max(words2))
	       words3=[w for w in words2 if w!=0]
               mstat4.append((float)(k)*100/len(xx))



plt.hold(False)  
z=plt.hist(mstat3,bins=[0,50,100,150,200,250,300,350,400],color='red')
plt.ylabel("Number of references")
plt.xlabel("Max Common Verb of the abstract with the text")
plt.ylim(0,max(z[0])+2)
plt.savefig("/home/raka/stats/mstat7.png")



plt.hold(False)  
z=plt.hist(mstat4,bins=8,color='red')
plt.ylabel("Number of references")
plt.xlabel("Percentage of Common Verb of abstract with the text")
plt.ylim(0,max(z[0])+2)
plt.savefig("/home/raka/stats/mstat8.png")


