import optparse

parser=optparse.OptionParser();

parser.add_option("--file_in", dest="file_in",
    action="store", type="string", default="", 
    help="file to transform")
parser.add_option("--pb_out", dest="pb_out",
    action="store", type="string", default="",
    help="the paberboat format")
parser.add_option("--memberships_out", dest="memberships_out",
    action="store", type="string", default="",
    help="the memberships")
(options, args)=parser.parse_args()

fin=open(options.file_in, "r")
fin.readline()
memberships={}
counter=0;
fout=open(options.pb_out, "w")
for line in fin:
  tokens=line.strip("\n").split(",")
  print >>fout, tokens[5]+","+tokens[4]
  if memberships.has_key(tokens[0])==False:
    memberships[tokens[0]]=[]
  memberships[tokens[0]].append(counter)
  counter+=1

fout=open(options.memberships_out, "w")
print >>fout,"header,sparse:int32:"+str(counter)
for elem in memberships.items():
  line="0,0,0,"
  for i in elem[1]:
    line+=str(i)+":1,"
  print >>fout, line

