#!/usr/bin/python
import sys
import optparse
import array

parser = optparse.OptionParser();
parser.add_option("--csv_file", action="store", type="string",
      dest="csv_file", default="",
      help="File that contains data in comma separated values (csv) format "
           "You can have comments starting with # in the begining of the file");

parser.add_option("--output_file", action="store", type="string",
      dest="output_file", default="",
      help="File that contains the converted data in the 1305 format");

(options, args)=parser.parse_args();

# now open the csv light file
fpin = open(options.csv_file, "r");

#open the file to write
fpout = open(options.output_file, "w");

# Read the 1st three lines and throw them out.
num_of_features = 0;
line_num = 0;
for line in fpin:
  line_num=line_num+1;
  
  # The 2nd line gives the number of features.
  if(line_num == 2):
    num_of_features = int(line);
  if (line_num == 3):
    break;

#start forming the header
header="header meta:3 sparse:double:"+str(num_of_features);

#write the header
print >> fpout, header;

# Compute the sum for each point.
normalizing_sums = array.array('l', []);
previous_point_id=-1;
new_sum = 0;
for line in fpin:

  #remove some trailing spaces
  line=line.rstrip('\n');
  line=line.rstrip('\r');
  line=line.rstrip(" ,");
  line=line.lstrip(" ,");
  tokens=line.split(" ");
  
  # Get the current point ID.
  current_point_id = int(tokens[0]);
  if(previous_point_id == -1):
    previous_point_id = current_point_id;

  # Compare it to the previous point ID and decide whether to write.
  if (current_point_id != previous_point_id):
    previous_point_id = current_point_id;
    normalizing_sums.append(new_sum);
    new_sum = 0;

  # Accumulate the sum.
  new_sum += int(tokens[2]);
  
# Append the last sum.
normalizing_sums.append(new_sum);

#now start the main loop for conversion
previous_point_id=-1;
outline=str(0) + "," + str(0) + "," + str(0) + ",";
point_num=0;
fpin.close();

fpin = open(options.csv_file, "r");
line_num = 0;
for line in fpin:
  line_num=line_num+1;
  
  # The 2nd line gives the number of features.
  if(line_num == 2):
    num_of_features = int(line);
  if (line_num == 3):
    break;

for line in fpin:

  #remove some trailing spaces
  line=line.rstrip('\n');
  line=line.rstrip('\r');
  line=line.rstrip(" ,");
  line=line.lstrip(" ,");
  tokens=line.split(" ");
  
  # Get the current point ID.
  current_point_id = int(tokens[0]);
  if(previous_point_id == -1):
    previous_point_id = current_point_id;

  # Compare it to the previous point ID and decide whether to write.
  if (current_point_id != previous_point_id):
    previous_point_id = current_point_id;
    outline=outline.rstrip(",");
    print >> fpout, outline;
    outline=str(0)+","+str(0)+","+str(0)+",";
    point_num+=1;

  # Concatenate the feature.
  new_value = float(tokens[2]) / normalizing_sums[point_num];
  if(new_value > 0):
    outline += str(int(tokens[1]) - 1) + ":" + str(new_value) + ",";
  
# Write the last line.
outline=outline.rstrip(",");
print >> fpout, outline;

# close the files 
fpin.close();
fpout.close();
