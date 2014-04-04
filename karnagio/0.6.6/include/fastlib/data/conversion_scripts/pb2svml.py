import scipy
import numpy
import scipy.sparse
import optparse

def Pb2Svml(filename_in, meta, filename_out):
  fin=open(filename_in, "r")
  header=fin.readline().strip(",\n")
  tokens=header.split(",")
  has_meta=False
  is_sparse=False
  dimensionality=0
  if tokens[1].startswith("meta"):
    has_meta=True

  for t in tokens[2:len(tokens)]:
    t1=t.split(":")
    if t1[0]=="sparse":
      is_sparse=True
      dimensionality+=int(t1[2])
    else:
      dimensionality+=int(t1[1])

  num_of_points=0
  for line in fin:
    num_of_points+=1
  fin=open(filename_in, "r")
  fin.readline()

  point_counter=0    
  fout=open(filename_out, "w")
  for line in fin:
    col_counter=0
    tokens=line.strip(",\n").split(",")
    start_column=0
    if has_meta:
      start_column=3
      print >> fout, tokens[meta],
    if is_sparse:
      for t in tokens[start_column:len(tokens)]:
        print >> fout, t,  
    else:
      for t in tokens[start_column:len(tokens)]:
        print >>fout, str(col_counter)+":"+t
        col_counter+=1
    point_counter+=1
    print >>fout,"\n",

if __name__=="__main__":
  parser = optparse.OptionParser();
  parser.add_option("--file_in", action="store", type="string",
      dest="file_in",
      help="file in paperboat format");

  parser.add_option("--file_out", action="store", type="string",
      dest="file_out",
      help="output file in svm light format");

  parser.add_option("--meta_index", action="store", type="string",
      dest="meta_index", default="1",
      help="the meta data index file to be used for target value, it can be 0 or 1 or 2");

  (options, args)=parser.parse_args();
  Pb2Svml(options.file_in, int(options.meta_index), options.file_out)
