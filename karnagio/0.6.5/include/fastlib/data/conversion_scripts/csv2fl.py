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

# this function returns the number of features in the first line of the file
def FindFeatures(filename):
  num_of_features=0;
  fpin=open(filename, "r");
  last_line="";
  for line in fpin:
    if line[0]!='\#':
      last_line=line;
      break;
    line=line.rstrip("\n");
    line=line.rstrip(" ,");
    line=line.lstrip(" ,");
  tokens=line.split(",");
  fpin.close();
  return len(tokens);

def MakeDictionaries(filename, indices):
  fpin=open(filename, "r");
  # get rid of the comments
  last_line="";
  last_position=fpin.tell();
  for line in fpin:
    if line[0]!='\#':
      last_line=line;
      break;
    last_position.tell();
  fpin.seek(last_position);
  dictionaries=[];
  for i in xrange(0, len(indices)):
    dictionaries.append({});
  for line in fpin:
    #remove some trailing spaces
    line=line.rstrip(" ,");
    line=line.rstrip("\n");
    line=line.lstrip(" ,");
    tokens=line.split(",");
    # now scan the tokens and create entries;
    for i in xrange(0, len(indices)):
      if (not dictionaries[i].has_key(tokens[indices[i]])):
        dictionaries[i][tokens[indices[i]]]=len(dictionaries[i]);
  fpin.close();
  # now create the offsets
  offsets=[0]*len(indices);
  for i in xrange(1, len(indices)):
    offsets[i]=offsets[i-1]+len(dictionaries[i-1]);
  # now compute all the categorical 0/1 attributes
  categorical_features=0;
  
  for i in dictionaries:
    categorical_features+=len(i);
  return (dictionaries, offsets, categorical_features)

parser = optparse.OptionParser();
parser.add_option("--csv_file", action="store", type="string",
      dest="csv_file", default="",
      help="File that contains data in comma separated values (csv) format "
           "You can have comments starting with # in the begining of the file");

parser.add_option("--output_file", action="store", type="string",
      dest="output_file", default="",
      help="File that contains the converted data in the 1305 format");

parser.add_option("--dense_double", action="store", type="string",
    dest="dense_double", default="",
    help="A comma separated list of integers refering to features that will "
         "be treated as continuous double precision data. "
         "If you want to include a range of indices use this syntax '1..10'. "
         "Note that we number columns starting at 0. "
         "In general the list can be mixed i.e: '1,4,3..8,99..101,167'. "
         "If you want to treat all data as dense double "
         "use '--dense_double=all' instead of a comma separated list. "
         "Note that you can only use the 'all' option once and no other "
         "options should be used when using 'all'. ");

parser.add_option("--dense_float", action="store", type="string",
    dest="dense_float", default="",
    help="Same as '--dense_double' except the data is now read and stored in floating point (single precision)");

parser.add_option("--dense_int", action="store", type="string",
    dest="dense_int", default="",
    help="Same as '--dense_double' except the data is now read and stored in integers.");

parser.add_option("--dense_bool", action="store", type="string",
    dest="dense_bool", default="",
    help="Same as '--dense_double' except the data is now read and stored in boolean.");

parser.add_option("--categorical", action="store", type="string",
    dest="categorical", default="",
    help="A comma separated list of integers refering to features that will "
         "be treated as categorical data. Syntax is similar to that for '--dense_double'. "
         "It will automatically detect the number of categories for each feature. "
         "If a feature has N different categories the feature will be expanded into "
         "N 0/1 features");
parser.add_option("--categorical2double", action="store", type="string",
    dest="categorical2double", default="",
    help="(Not currently supporte) A comma separated list of integers refering to features that will "
         "be treated as categorical data, but stored as double. Syntax is similar to that for '--dense_double'. "
         "It will automatically detect the number of categories for each feature. "
         "If a feature has N different categories the feature will be expanded into "
         "N 0/1 features");

parser.add_option("--meta_label", action="store", type="string",
    dest="meta_label", default="",
    help="The index of the column that will be treated as a label "
          "if it is a two class problem it will automatically be converted "
          "to -1, +1, otherwise it will be converted to the positive integers"
          "The cardinality will be equal to the number of classes");

parser.add_option("--meta_target", action="store", type="string",
    dest="meta_target", default="",
    help="The index of the column that will be treated as a target value "
         "for the point. It can be any double number. This is useful for "
         "Regression or other predictive algorithms. Values remain as is.");
  
parser.add_option("--meta_id", action="store", type="string",
    dest="meta_id", default="",
    help="The index from the column that will be treated as a unique id "
         "for the point. It should be an integer. Presently no uniqueness "
         "checks are done. Please insure uniqueness in input file");


(options, args)=parser.parse_args();
dense_precisions=[];
dense_features=[];
categorical_indices=[];
all_dense=0;
all_sparse=0;
# get the number of features
num_of_features=FindFeatures(options.csv_file);
#start forming the header
header="header,meta:3,";

if len(options.dense_bool):
  dense_precisions.append("bool");
  if (options.dense_bool=="all"):
    all_dense=1;
    header+="bool:"+str(num_of_features)+",";
  else:
    dense_features.append(ParseIndices(options.dense_bool));
    header+="bool:"+str(len(dense_features[-1]))+",";

if len(options.dense_int):
  if (all_dense==1):
      print "You have set more than one precision flags to \"all\" ";
      exit();
  dense_precisions.append("int");
  if (options.dense_int=="all"):
    all_dense=1;
    header+="int:"+str(num_of_features)+",";
  else:
    dense_features.append(ParseIndices(options.dense_int));
    header+="int:"+str(len(dense_features[-1]))+",";

if len(options.dense_float):
  if (all_dense==1):
    print "You have set more than one precision flags to \"all\" ";
    exit();
  dense_precisions.append("float");
  if (options.dense_float=="all"):
    all_dense=1;
    header+="float:"+str(num_of_features)+",";
  else:
    dense_features.append(ParseIndices(options.dense_float));
    header+="float:"+str(len(dense_features[-1]))+",";

if len(options.dense_double):
  if (all_dense==1):
    print "You have set more than one precision flags to \"all\" ";
    exit();
  dense_precisions.append("double");
  if (options.dense_double=="all"):
    all_dense=1;
    header+="double:"+str(num_of_features)+",";
  else:
    dense_features.append(ParseIndices(options.dense_double));
    header+="double:"+str(len(dense_features[-1]))+",";

dense_offset=0;
for k in dense_features:
  dense_offset+=len(k);


if len(options.categorical):
  if (all_dense==1):
    print "You have set more than one precision flags to \"all\" ";
    exit();
  categorical_indices=ParseIndices(options.categorical); 
  # if all sparse
  if (categorical_indices==[]):
    all_sparse=1;
    # categorical_indices = range(0,num_of_features);
    # prevent meta info from being considered
    for k in xrange(0, num_of_features):
      if(str(k) <> options.meta_label and str(k) <> options.meta_target and str(k) <> options.meta_id) :
        categorical_indices.append(k);
  (dictionaries, cat_offsets, total_bool_features) = MakeDictionaries(options.csv_file, categorical_indices);
  header+="sparse:bool:"+str(total_bool_features)+",";

# now open the csv light file
fpin = open(options.csv_file, "r");
#open the file to write
fpout = open(options.output_file, "w");
#open config file
fp_config = open("conversion_config.txt", "w");
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

if(all_sparse==1 or all_dense==1):
  if (options.meta_label!=""):
    print "Info: Even though 'all' options is used I am ignoring column " + str(options.meta_label) + " and using it as the label as specified in option '--meta_label'"
  if (options.meta_target!=""):
    print "Info: Even though 'all' options is used I am ignoring column " + str(options.meta_target) + " and using it as the target as specified in option '--meta_target'"
  if (options.meta_id!=""):
    print "Info: Even though 'all' options is used I am ignoring column " + str(options.meta_id) + " and using it as the id as specified in option '--meta_id'"

for line in fpin:
  #remove some trailing spaces
  line=line.rstrip('\n');
  line=line.rstrip('\r');
  line=line.rstrip(" ,");
  line=line.lstrip(" ,");
  tokens=line.split(",");
  # Work on the metadata now
  meta1=0;
  meta2=0;
  meta3=0;
  if (options.meta_label!=""):
    meta1=tokens[int(options.meta_label)];
  if (options.meta_target!=""):
    meta2=tokens[int(options.meta_taget)];
  if (options.meta_id!=""):
    meta3=tokens[int(options.meta_id)];
  # now we form the output line
  outline=str(meta1)+","+str(meta2)+","+str(meta3)+",";
  if (all_dense==1):
    i = 0;
    for tok in tokens:
      if(str(i) <> options.meta_label and str(i) <> options.meta_target and str(i) <> options.meta_id) :
        outline+=tok+",";
      i+=1
  else:
    for k in xrange(0, len(dense_precisions)):
      for i in dense_features[k]:
        outline+=tokens[i]+",";
  
  # now write the categorical
  for i in xrange(0, len(categorical_indices)):
    ind=dense_offset+cat_offsets[i]+dictionaries[i][tokens[categorical_indices[i]]];
    outline+=str(ind)+":1,";
    
  outline=outline.rstrip(",");
  print >> fpout, outline;
  
# close the files 
fpin.close();
fpout.close();
fp_config.close();

