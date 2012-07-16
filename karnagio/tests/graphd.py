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
targets="graphd"
dataset_dir="./datasets"
test_dir=os.getcwd()

# Build targets
test_suite.Build(version_dir, test_dir, targets, fout);

# start the wtachdog timer that will stop the test in case
# something goes wrong with the tests and they deadlock
t=threading.Timer(120, test_suite.ExpireTest, "graphd", fout)
       
# Start the tests
for directory in bin_dir.values():
  graphd1=directory+"/graphd --references_in="+     \
      dataset_dir+"/random/random_1kx6.txt "+       \
      " --symmetric_diffusion=0"                    \
      +" --left_labels_in="                         \
      +dataset_dir+"/random/left_random_1k.csv"     \
      +" --right_labels_out=right_labels"           \
      +" --diffusion_iterations=10"                 \
      +" --left_labels_out=left_labels"             \
      +" --right_labels_norm=l2"                    \
      +" --left_labels_norm=l1"                     \
      +" --connect_nodes=snn"                       \
      +" --graph_out=graph"                         \
      +" --weight_policy=\"exp(-dist/h)\""          \
      +" --run_diffusion=1"                         \
      +" --summary=svd"                             \
      +" --nn:k_neighbors=5"                        \
      +" --svd:svd_rank=2"

  os.system(graphd1 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, graphd1, "SUCCESS"
  else:
    print >> fout, graphd1, "FAILED"
    print graphd1, "FAILED"
  os.remove("temp")
  if os.path.exists("graph"):
    os.remove("graph")
  else:
    print >> fout, graphd1, "(graph) FAILED"
  if os.path.exists("left_labels"):
    os.remove("left_labels")
  else:
    print >> fout, graphd1, "(left_labels) FAILED"
  if os.path.exists("right_labels"):
    os.remove("right_labels")
  else:
    print >> fout, graphd1, "(right_labels) FAILED"
 
  graphd2=directory+"/graphd --references_in="+     \
      dataset_dir+"/random/random_1kx6.txt "+       \
      " --symmetric_diffusion=1"                    \
      +" --right_labels_in="                        \
      +dataset_dir+"/random/right_random_1k.csv"    \
      +" --right_labels_out=right_labels"           \
      +" --diffusion_iterations=10"                 \
      +" --right_labels_norm=l2"                    \
      +" --left_labels_norm=l1"                     \
      +" --connect_nodes=snn"                       \
      +" --graph_out=graph"                         \
      +" --weight_policy=\"exp(-dist/h)\""          \
      +" --run_diffusion=1"                         \
      +" --summary=svd"                             \
      +" --nn:k_neighbors=5"                        \
      +" --svd:svd_rank=2"

  os.system(graphd2 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, graphd2, "SUCCESS"
  else:
    print >> fout, graphd2, "FAILED"
    print graphd2, "FAILED"
  os.remove("temp")
  if os.path.exists("graph"):
    os.remove("graph")
  else:
    print >> fout, graphd2, "(graph) FAILED"
  if os.path.exists("right_labels"):
    os.remove("right_labels")
  else:
    print >> fout, graphd2, "(right_labels) FAILED"
 
  graphd3=directory+"/graphd --references_in="+     \
      dataset_dir+"/random/random_1kx6.txt "+       \
      " --symmetric_diffusion=0"                    \
      +" --left_labels_in="                         \
      +dataset_dir+"/random/left_random_1k.csv"     \
      +" --right_labels_out=right_labels"           \
      +" --diffusion_iterations=10"                 \
      +" --right_labels_norm=l2"                    \
      +" --connect_nodes=snn"                       \
      +" --graph_out=graph"                         \
      +" --weight_policy=\"exp(-dist/h)\""          \
      +" --run_diffusion=1"                         \
      +" --summary=svd"                             \
      +" --nn:k_neighbors=5"                        \
      +" --svd:svd_rank=2"

  os.system(graphd3 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, graphd3, "SUCCESS"
  else:
    print >> fout, graphd3, "FAILED"
    print graphd3, "FAILED"
  os.remove("temp")
  if os.path.exists("graph"):
    os.remove("graph")
  else:
    print >> fout, graphd3, "(graph) FAILED"
  if os.path.exists("right_labels"):
    os.remove("right_labels")
  else:
    print >> fout, graphd3, "(right_labels) FAILED"
 
  graphd3=directory+"/graphd --references_in="+     \
      dataset_dir+"/random/random_1kx6.txt "+       \
      " --symmetric_diffusion=0"                    \
      +" --left_labels_in="                         \
      +dataset_dir+"/random/left_random_1k.csv"     \
      +" --right_labels_out=right_labels"           \
      +" --diffusion_iterations=10"                 \
      +" --right_labels_norm=l2"                    \
      +" --connect_nodes=snn"                       \
      +" --graph_out=graph"                         \
      +" --weight_policy=\"exp(-dist/h)\""          \
      +" --run_diffusion=1"                         \
      +" --summary=svd"                             \
      +" --nn:k_neighbors=5"                        \
      +" --svd:svd_rank=2"

  os.system(graphd3 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, graphd3, "SUCCESS"
  else:
    print >> fout, graphd3, "FAILED"
    print graphd3, "FAILED"
  os.remove("temp")
  if os.path.exists("graph"):
    os.remove("graph")
  else:
    print >> fout, graphd3, "(graph) FAILED"
  if os.path.exists("right_labels"):
    os.remove("right_labels")
  else:
    print >> fout, graphd3, "(right_labels) FAILED"
 
  graphd4=directory+"/graphd --references_in="+     \
      dataset_dir+"/random/random_1kx6.txt "+       \
      " --symmetric_diffusion=0"                    \
      +" --left_labels_in="                         \
      +dataset_dir+"/random/left_random_1k.csv"     \
      +" --right_labels_out=right_labels"           \
      +" --diffusion_iterations=10"                 \
      +" --right_labels_norm=l2"                    \
      +" --connect_nodes=snn"                       \
      +" --graph_out=graph"                         \
      +" --weight_policy=\"exp(-dist/h)\""          \
      +" --run_diffusion=1"                         \
      +" --summary=none"                            \
      +" --nn:k_neighbors=5"                        \
     

  os.system(graphd4 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, graphd4, "SUCCESS"
  else:
    print >> fout, graphd4, "FAILED"
    print graphd4, "FAILED"
  os.remove("temp")
  if os.path.exists("graph"):
    os.remove("graph")
  else:
    print >> fout, graphd4, "(graph) FAILED"
  if os.path.exists("right_labels"):
    os.remove("right_labels")
  else:
    print >> fout, graphd4, "(right_labels) FAILED"
 
  graphd5=directory+"/graphd --references_in="+     \
      dataset_dir+"/random/random_1kx6.txt "+       \
      " --symmetric_diffusion=0"                    \
      +" --left_labels_in="                         \
      +dataset_dir+"/random/left_random_1k.csv"     \
      +" --right_labels_out=right_labels"           \
      +" --diffusion_iterations=10"                 \
      +" --right_labels_norm=l2"                    \
      +" --weight_policy=1/dist"                    \
      +" --run_diffusion=1"                         \
      +" --summary=none"                            \
     

  os.system(graphd5 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, graphd5, "SUCCESS"
  else:
    print >> fout, graphd4, "FAILED"
    print graphd5, "FAILED"
  os.remove("temp")
  if os.path.exists("right_labels"):
    os.remove("right_labels")
  else:
    print >> fout, graphd4, "(right_labels) FAILED"
 
