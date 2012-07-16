import os
import os.path
import threading
import test_suite
import optparse

parser = optparse.OptionParser();

parser.add_option("--version_dir", action="store", type="string", 
    dest="version_dir", default="../branches/0.6.4/", 
    help="the version directory to build/use the binaries")
parser.add_option("--logfile", action="store", type="string", 
    dest="logfile", default="test_results",
    help="The file to append the results")
(options, args)=parser.parse_args();

# open log file for appending output
fout=open(options.logfile, "a")

version_dir=options.version_dir
build_mode=["config-debug"]
bin_dir={"debug":(version_dir+"debug.build/bin"), \
    "release":(version_dir+"release.build/bin")}
targets="allkn"
dataset_dir="./datasets"
test_dir=os.getcwd()

# Build targets
test_suite.Build(version_dir, test_dir, targets, fout);

# start the wtachdog timer that will stop the test in case
# something goes wrong with the tests and they deadlock
t=threading.Timer(120, test_suite.ExpireTest, "allkn", fout)

# Start the tests
for directory in bin_dir.values():
  allkn1=directory+"/allkn --references_in="+\
      dataset_dir+"/random/random_1kx6.txt "+       \
      " --k_neighbors=5";
  os.system(allkn1 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, allkn1, "SUCCESS"
  else:
    print >> fout, allkn1, "FAILED"
    print allkn1, "FAILED"
  os.remove("temp")
  
  allkn2=directory+"/allkn --references_in="+ \
         dataset_dir+"/random/random_1kx6.txt "+     \
         " --k_neighbors=5 "+                  \
         " --distances_out=distances.txt "+    \
         " --indices_out=indices.txt"
  os.system(allkn2 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", ["distances.txt", "indices.txt"], []))==True:
    print >> fout, allkn2, "SUCCESS"
  else:
    print >> fout, allkn2, "FAILED"
    print allkn2, "FAILED"

  os.remove("temp")
  if os.path.exists("distances.txt")==True:
    os.remove("distances.txt")
  if os.path.exists("indices.txt")==True:  
    os.remove("indices.txt")
  
  allkn3=directory+"/allkn "+                                 \
         " --references_in="+dataset_dir+"/bag_of_words/docword_nips.txt " \
         " --k_neighbors=1 --distances_out=distances.txt --indices_out=indices.txt"
  os.system(allkn3 + " 2>&1 >temp")
  if (test_suite.EvaluateRun("temp", ["distances.txt", "indices.txt"], []))==True:
    print >> fout, allkn3, "SUCCESS"
  else:
    print >> fout, allkn3, "FAILED"
    print allkn3, "FAILED"

  os.remove("temp")
  if os.path.exists("distances.txt")==True:
    os.remove("distances.txt")
  if os.path.exists("indices.txt")==True:  
    os.remove("indices.txt") 

  allkn4=directory+"/allkn "+                               \
         " --references_in="+dataset_dir+"/random/random_1kx6.txt "\
         " --k_neighbors=5 "+                               \
         " --queries_in=" +dataset_dir+"/random/random_1kx6.txt " \
         " --distances_out=distances.txt --indices_out=indices.txt"
  os.system(allkn4 + " 2>&1 >temp")
  if (test_suite.EvaluateRun("temp", ["distances.txt", "indices.txt"], []))==True:
    print >> fout, allkn4, "SUCCESS"
  else:
    print >> fout, allkn4, "FAILED"
    print allkn4, "FAILED"

  os.remove("temp")
  if os.path.exists("distances.txt")==True:
    os.remove("distances.txt")
  if os.path.exists("indices.txt")==True:  
    os.remove("indices.txt")

  allkn5=directory+"/allkn --references_in="+              \
         dataset_dir+"/random/random_1kx6.txt "+                   \
         " --k_neighbors=2  "+                              \
         " --method=furthest " \
         " --distances_out=distances.txt --indices_out=indices.txt"
  os.system(allkn5 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", ["distances.txt", "indices.txt"], []))==True:
    print >> fout, allkn5, "SUCCESS"
  else:
    print >> fout, allkn5, "FAILED"
    print allkn5, "FAILED"

  os.remove("temp")
  if os.path.exists("distances.txt")==True:
    os.remove("distances.txt")
  if os.path.exists("indices.txt")==True:  
    os.remove("indices.txt")

 
  allkn6=directory+"/allkn  --references_in="+              \
      dataset_dir+"/random/random_1kx6.txt"+                \
      " --r_neighbors=0.002 "                               \
      " --distances_out=distances.txt --indices_out=indices.txt"
  os.system(allkn6 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", ["distances.txt", "indices.txt"], []))==True:
    print >> fout, allkn6, "SUCCESS"
  else:
    print >> fout, allkn6, "FAILED"
    print allkn6, "FAILED"

  os.remove("temp")
  if os.path.exists("distances.txt")==True:
    os.remove("distances.txt")
  if os.path.exists("indices.txt")==True:  
    os.remove("indices.txt")


print >> fout, "[allkn] Test finished"
fout.close()
t.cancel()
