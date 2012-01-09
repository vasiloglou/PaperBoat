#!/usr/bin/python
import sys
import optparse

# these are functions that we need for the script
def ParseIndices(token):
 indices=[];
 toks=token.split(",");
 for t in toks:
   # check if it is all
   if (t=="all"):
     return [];
   # see if this token is of the form a..b
   if (t.find("..")>0):
     (a, b)=t.split("..");
     for i in xrange(int(a), int(b)+1):
       indices.append(i);
   else:
     indices.append(int(t));
 return indices;


def FindFeatures(filename):
  num_of_features=0;
  fpin=open(filename, "r");
  last_line="";
  for line in fpin:
    if line[0]!='\#':
      last_line=line;
      break;
  #find the comment in the end of the line and ignore it
  ind = line.find('\#');
  if (ind!=-1):
    line=line[0:ind];
  tokens=line.split(" ");
  (index, value)=tokens[-1].split(":");
  if (index > num_of_features):
    num_of_features=index;
  # now continue with the other lines
  for line in fpin:
    #find the comment in the end of the line and ignore it
    ind = line.find('\#');
    if (ind!=-1):
      line=line[0:ind];
    tokens=line.split(" ");
    (index, value)=tokens[-1].split(":");
    if (index > num_of_features):
      num_of_features=index;
  fpin.close();
  return num_of_features;

parser = optparse.OptionParser();
parser.add_option("--svmlight_file", action="store", type="string",
      dest="svmlight_file", default="",
      help="File that contains data in svm light format");

parser.add_option("--output_file", action="store", type="string",
      dest="output_file", default="",
      help="File that contains the converted data in the 1305 format");

parser.add_option("--dense_double", action="store", type="string",
    dest="dense_double", default="",
    help="A comma separated list of integers refering to features that will "
         "be treated as dense double data. "
         "If you want to include a range of indices use this syntax 1..10"
         "In general the list can be mixed ie: =1,4,3..8,99..101,167"
         "If you want to treat all data as dense double "
         "use \"all\" instead of a comma separated list");

parser.add_option("--dense_float", action="store", type="string",
    dest="dense_float", default="",
    help="A comma separated list of integers refering to features that will "
         "be treated as dense float data."
         "If you want to include a range of indices use this syntax 1..10"
         "In general the list can be mixed ie: =1,4,3..8,99..101,167"
         "If you want to treat all data as dense float "
         "use \"all\" instead of a comma separated list");

parser.add_option("--dense_int", action="store", type="string",
    dest="dense_int", default="",
    help="A comma separated list of integers refering to features that will "
         "be treated as dense int data. "
         "If you want to include a range of indices use this syntax 1..10"
         "In general the list can be mixed ie: =1,4,3..8,99..101,167"
         "If you want to treat all data as dense int "
         "use \"all\" instead of a comma separated list");

parser.add_option("--dense_bool", action="store", type="string",
    dest="dense_bool", default="",
    help="A comma separated list of integers refering to features that will "
         "be treated as dense bool data. "
         "If you want to include a range of indices use this syntax 1..10"
         "In general the list can be mixed ie: =1,4,3..8,99..101,167"
         "If you want to treat all data as dense bool "
         "use \"all\" instead of a comma separated list");

parser.add_option("--sparse_double", action="store", type="string",
    dest="sparse_double", default="",
    help="A comma separated list of integers refering to features that will "
         "be treated as sparse double data. "
         "If you want to include a range of indices use this syntax 1..10"
         "In general the list can be mixed ie: =1,4,3..8,99..101,167"
         "If you want to treat all data as sparse double "
         "use \"all\" instead of a comma separated list");

parser.add_option("--sparse_float", action="store", type="string",
    dest="sparse_float", default="",
    help="A comma separated list of integers refering to features that will "
         "be treated as sparse float data. "
         "If you want to include a range of indices use this syntax 1..10"
         "In general the list can be mixed ie: =1,4,3..8,99..101,167"
         "If you want to treat all data as sparse float "
         "use \"all\" instead of a comma separated list");

parser.add_option("--sparse_int", action="store", type="string",
    dest="sparse_int", default="",
    help="A comma separated list of integers refering to features that will "
         "be treated as sparse int data. "
         "If you want to include a range of indices use this syntax 1..10"
         "In general the list can be mixed ie: =1,4,3..8,99..101,167"
         "If you want to treat all data as sparse int "
         "use \"all\" instead of a comma separated list");

parser.add_option("--sparse_bool", action="store", type="string",
    dest="sparse_bool", default="",
    help="A comma separated list of integers refering to features that will "
         "be treated as sparse bool data. "
         "If you want to include a range of indices use this syntax 1..10"
         "In general the list can be mixed ie: =1,4,3..8,99..101,167"
         "If you want to treat all data as sparse bool "
         "use \"all\" instead of a comma separated list");



(options, args)=parser.parse_args();
dense_precisions=[];
dense_features=[];
sparse_precisions=[];
sparse_features=[];
all_dense=0;
all_sparse=0;
# get the number of features
num_of_features=FindFeatures(options.svmlight_file);
#start forming the header
header="header meta:3 ";

if len(options.dense_bool):
  dense_precisions.append("bool");
  if (options.dense_bool=="all"):
    all_dense=1;
    header+="dense:bool:"+str(num_of_features)+" ";
  dense_features.append(ParseIndices(options.dense_bool));
  header+="dense:bool:"+str(len(dense_features[-1]))+" ";

if len(options.dense_int):
  dense_precisions.append("int");
  if (options.dense_int=="all"):
    if (all_dense==1):
      print "You have set more than one precision flags to \"all\" ";
      exit();
    all_dense=1;
    header+="dense:int:"+str(num_of_features)+" ";
  else:
    dense_features.append(ParseIndices(options.dense_int));
    header+="dense:int:"+str(len(dense_features[-1]))+" ";

if len(options.dense_float):
  dense_precisions.append("float");
  if (options.dense_float=="all"):
    if (all_dense==1):
      print "You have set more than one precision flags to \"all\" ";
      exit();
    all_dense=1;
    header+="dense:float:"+str(num_of_features)+" ";
  else:
    dense_features.append(ParseIndices(options.dense_float));
    header+="dense:float:"+str(len(dense_features[-1]))+" ";

if len(options.dense_double):
  dense_precisions.append("double");
  if (options.dense_double=="all"):
    if (all_dense==1):
      print "You have set more than one precision flags to \"all\" ";
      exit();
    all_dense=1;
    header+="dense:double:"+str(num_of_features)+" ";
  else:
    dense_features.append(ParseIndices(options.dense_double));
    header+="dense:double:"+str(len(dense_features[-1]))+" ";


if len(options.sparse_bool):
  sparse_precisions.append("bool");
  if (options.sparse_bool=="all"):
    if (all_dense==1):
      print "You have set more than one precision flags to \"all\" ";
      exit();
    all_sparse=1;
    header+="sparse:bool:"+str(num_of_features)+" ";
  else:
    sparse_features.append(ParseIndices(options.sparse_bool));
    header+="sparse:bool:"+str(len(sparse_features[-1]))+" ";

if len(options.sparse_int):
  sparse_precisions.append("int");
  if (options.sparse_int=="all"):
    if (all_dense==1 or all_sparse==1):
      print "You have set more than one precision flags to \"all\" ";
      exit();
    all_sparse=1;
    header+="sparse:int:"+str(num_of_features)+" ";
  else:
    sparse_features.append(ParseIndices(options.sparse_int));
    header+="sparse:int:"+str(len(sparse_features[-1]))+" ";

if len(options.sparse_float):
  sparse_precisions.append("float");
  if (options.sparse_float=="all"):
    if (all_dense==1 or all_sparse==1):
      print "You have set more than one precision flags to \"all\" ";
      exit();
    all_sparse=1;
    header+="sparse:float:"+str(num_of_features)+" ";
  else:
    sparse_features.append(ParseIndices(options.sparse_float));
    header+="sparse:float:"+str(len(sparse_features[-1]))+" ";

if len(options.sparse_double):
  sparse_precisions.append("double");
  if (options.sparse_double=="all"):
    if (all_dense==1 or all_sparse==1):
      print "You have set more than one precision flags to \"all\" ";
      exit();
    all_sparse=1;
    header+="sparse:double:"+num_of_features+" ";
  else:
    sparse_features.append(ParseIndices(options.sparse_double));
    header+="sparse:double:"+str(len(sparse_features[-1]))+" ";

if (len(sparse_precisions)>1 or len(dense_precisions)>1):
  print "Warning!!! Although you can convert the file to multiple precisions "
  "The current version of 1305 might not support this option"

# now open the svm light file
fpin = open(options.svmlight_file, "r");
#open the file to write
fpout = open(options.output_file, "w");
#write the header
print >> fpout, header;

#now start the main loop for conversion
#ignore comments first
last_line="";
last_position=fpin.tell();
for line in fpin:
  if line[0]!='\#':
    last_line=line;
    break;
  last_position.tell();

fpin.seek(last_position);

for line in fpin:
  #find the comment in the end of the line and ignore it
  ind = line.find('\#');
  if (ind!=-1):
    line=line[0:ind];
  #remove some trailing spaces
  line=line.rstrip();
  line=line.lstrip();
  tokens=line.split(" ");
  # Work on the metadata now
  meta1=0;
  meta2=0;
  meta3=0;
  # the first one is the label or target
  if (abs(int(tokens[0])) <=1):
    #in this case it is a class label and we store it in the
    #first metadata slot
    meta1=int(tokens[0]);
  else:
    #otherwise it is a regression target value and we store it in 
    #the second slot
    meta2=float(tokens[0]);

  tokens.pop(0);
  #now we have to search if there is any qid token 
  (index, value) = tokens[0].split(":");
  if (index=="qid"):
    meta3=int(value);
    tokens.pop(0); 

  # now we form the output line
  outline=str(meta1)+" "+str(meta2)+" "+str(meta3)+" ";
  if (all_dense==1):
    for tok in tokens:
      (index, value)=tok.split(":");
      outline+=value+" ";
  else:
    if (all_sparse==1):
      for tok in tokens:
        outline+=tok+" ";
    else:
      # put everything in a dictionary
      dict={};
      for tok in tokens:
        (index, value)=tok.split(":");
        dict[int(index)]=value;
      #this is a current index
      ind=0;
      # now write out the dense  
      if (len(dense_precisions)>1):
        for k in xrange(0, len(dense_precisions)):
          for i in dense_features[k]:
            if (dict.has_key(i)):
              outline+=dense_precisions[k]+":"+dict[i]+" ";
            else:
              outline+=dense_precisions[k]+":0 ";
            ind+=1;
      else:
        for i in dense_features[0]:
          if (dict.has_key(i)):
            outline+=dict[i]+" ";
          else:
            outline+="0 ";
          ind+=1;
       # now write out the sparse 
      if (len(sparse_precisions)>1):
        for k in xrange(0, len(sparse_precisions)):
          for i in sparse_features[k]:
            if (dict.has_key(i)):
              outline+=sparse+":"+sparse_precisions[k]+":"+str(ind)+":"+dict[i]+" ";
            ind+=1;
      else:
        for i in sparse_features[0]:
          if (dict.has_key(i)):
            outline+=sparse_precisions[0]+":"+str(ind)+":"+dict[i]+" ";
          ind+=1;   
  print >> fpout, outline;
  
# close the files 
fpin.close();
fpout.close();





