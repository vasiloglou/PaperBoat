import os
import optparse

parser =optparse.OptionParser()
parser.add_option("--destination", action="store", type="string", 
    dest="destination")

(options, args)=parser.parse_args()
os.system("rm -f `find . -name *.swp` `find . -name *.swo` `find . -name *~`")
action="rsync -vr --exclude=\"*.~\" --exclude=\"*.swo\" --exclude=\"*.swp\" include src build_dependencies.py Makefile "+options.destination
os.system(action)

