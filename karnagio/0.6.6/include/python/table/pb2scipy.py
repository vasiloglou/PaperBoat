import scipy
import numpy
import scipy.sparse

def Pb2Scipy(filename, meta=[]):
  int8=[]
  float32=[]
  int64=[]
  precision=scipy.float64
  fin=open(filename, "r")
  header=fin.readline().strip("\n")
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

    if t1[1]=="double" or t1[0]=="double":
      precision=scipy.float64
      break
    else:
      if t1[1]=="uint8" or t1[0]=="uint8":
        precision=scipy.uint8
        break
      else:
        if t1[1]=="bool" or t1[0]=="bool":
          precision=scipy.bool8
          break
        else:
          if t1[1]=="int32" or t1[0]=="int32":
            precision=scipy.int32
            break
          else:
            if t1[1]=="uint32" or t1[0]=="uint32":
              precision=scipy.uint32
              break
            else:
              if t1[1]=="int64" or t1[0]=="int64":
                precision=scipy.int64
                break
              else:
                if t1[1]=="uint64" or t1[0]=="uint64":
                  precision=scipy.uint64
                  break
  num_of_points=0
  for line in fin:
    num_of_points+=1
  fin=open(filename, "r")
  fin.readline()

  if is_sparse:
    mat=scipy.sparse.lil_matrix((num_of_points, dimensionality),dtype=precision)
  else:
    mat=scipy.zeros((num_of_points, dimensionality), dtype=precision)
  point_counter=0    
  for line in fin:
    col_counter=0
    tokens=line.strip(",\n").split(",")
    start_column=0
    if has_meta:
      start_column=3
      for m in meta:
        if m==0:
          int8.append(int(tokens[0]))
        else:
          if m==1:
            float32.append(float(tokens[1]))
          else:
            int64.append(int(tokens[2]))
    if is_sparse:
      for t in tokens[start_column:len(tokens)]:
        (ind,val)=t.split(":")
        mat[point_counter, int(ind)]=precision(val)
    else:
      for t in tokens[start_column:len(tokens)]:
        mat[point_counter, col_counter]=precision(t)
        col_counter+=1
    point_counter+=1
  result=[mat]
  if 0 in meta:
    result.append(int8)
  if 1 in meta:
    result.append(float32)
  if 2 in meta:
    result.append(int64)
  return result


def Scipy2Pb(mat, filename):
  fout=open(filename, "w")
  if mat.__class__==numpy.ndarray:
    print >>fout,"header,meta:3,double:"+str(mat.shape[1])
    for i in range(mat.shape[0]):
      new_line="0,0,0,"
      for j in range(mat.shape[1]):
        new_line+=str(mat[i,j])+","
      print >>fout, new_line
  else:
    if mat.__class__==scipy.sparse.lil.lil_matrix:
      print >>fout,"header,meta:3,double:"+str(mat.shape[1])
      for i in range(mat.shape[1]):
        new_line="0,0,0,"
        for m in mat.getrow(i):
          new_line+=str(t[0][1])+":"+str(t[1])+","
        print >>fout, new_line
    else:
      print "I can only convert dense or scipy.sparse.lil.lil_matrix"




