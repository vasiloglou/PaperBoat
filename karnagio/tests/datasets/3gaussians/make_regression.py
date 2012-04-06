import scipy

fin1=open("3gaussians.csv", "r")
fout=open("3gaussians_regression", "w")
print >>fout, "header,meta:3,double:3"
i=0
for line1 in fin1:
  line1=line1.strip("\n")
  print >> fout,"0,"+str(i/30000+scipy.rand())+",0,"+line1+","
  i+=1

 
