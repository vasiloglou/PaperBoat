import os
import optparse

parser=optparse.OptionParser()

parser.add_option("--mode", action="store", type="string", 
    dest="mode", default="debug")
parser.add_option("--j", action="store", type="string",
    dest="j", default="7")
(options, args)=parser.parse_args()

command=""
if options.mode=="debug":
  command+="cd debug; "
  command+="rm -rf *; "
  command+="cmake -DCMAKE_BUILD_TYPE=Debug ../../src/pb-dev; "
  command+="make -j"+options.j+" paperboat"
  os.system("mkdir -p debug")
  os.system(command)
else:
  if options.mode=="release":
    command+="cd release; "
    command+="rm -rf *; "
    command+="cmake -DCMAKE_BUILD_TYPE=Release ../../src/pb-dev; "
    command+="make -j"+options.j+" paperboat"
    os.system("mkdir -p release")
    os.system(command)
  else:
    print "Mode ("+options.mode+") not recognized, nothing to be dome" 
