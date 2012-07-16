import scipy 

fout1=open("training.pb", "w");
fout2=open("augmented.pb", "w");

for j in range(10):
  for k in range(100):
    line=""
    print >>fout2, str(k)+","+str(j)+","
    for i in range(10):
      line+=str(i+(k % 10)+scipy.random.uniform(0, 0.5))+","
    print >>fout1, line



