fin=open("3gaussians.csv", "r")
fout=open("3gaussians.thor", "w")

line_num=0
file_id=0;
for line in fin:
  tokens=line.strip("\n").split(",")
  dim=0;
  for t in tokens:
    print >>fout, str(line_num)+","+str(dim)+","+t+","+str(file_id)
    dim+=1
  line_num+=1

