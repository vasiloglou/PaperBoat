import scipy

inds=[]
for i in range(50):
  inds.append(i)

for i in range(10):
  fout=open("tensor_sparse_100x50x3_"+str(i), "w")
  print >>fout, "header,meta:3,sparse:double:50,"
  for j in range(100):
    line="0,0,0,"
    new_inds=scipy.random.permutation(inds)[0:10]
    new_inds.sort()
    for k in range(10):
      ind=new_inds[k]
      val=scipy.rand()
      line+=str(ind)+":"+str(val)+","
    print >>fout, line
