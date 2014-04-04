import os

os.system("mkdir -p repos")
os.system("cd repos && wget http://www.bzip.org/1.0.6/bzip2-1.0.6.tar.gz ")
os.system("cd repos && tar -xzf bzip2-1.0.6.tar.gz ")
os.system("cd repos/bzip2-1.0.6/ && make all")
os.system("cd repos && wget http://zlib.net/zlib-1.2.8.tar.gz ")
os.system("cd repos && tar -xzf zlib-1.2.8.tar.gz ")
os.system("cd repos/zlib-1.2.8 && ./configure && make all")
os.system("cd repos && wget http://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.gz/download ")
os.system("cd repos && mv download boost-1.55.0.tar.gz")
os.system("cd repos && tar -xzf boost*.tar.gz")

action="cd repos/ && "\
    +" export BZIP2_LIBPATH=`pwd`/bzip2-1.0.6/ && export BZIP2_INCLUDE=`pwd`/bzip2-1.0.6/ && "\
    +" export BZIP2_SOURCE=`pwd`/bzip2-1.0.6/ && export BZIP2_BINARY=`pwd`/bzip2-1.0.6/ && "\
    +" export ZLIB_LIBPATH=`pwd`/zlib-1.2.8/ && export ZLIB_INCLUDE=`pwd`/zlib-1.2.8/ && "\
    +" export ZLIB_SOURCE=`pwd`/zlib-1.2.8/ && export ZLIB_BINARY=`pwd`/zlib-1.2.8/ && "\
    +"cd boost*/ && "\
    +"./bootstrap.sh"\
    +" --with-libraries="\
    +"program_options,"\
    +"serialization,"\
    +"system,filesystem,"\
    +"iostreams,"\
    +"thread "\
    +" && "\
    +"./b2 link=static "
print action
os.system(action)

os.system("cd repos && wget http://www.netlib.org/lapack/lapack-3.4.2.tgz ")
os.system("cd repos && tar -xzf lapack*.tgz")

os.system("cd repos && wget http://www.cmake.org/files/v2.8/cmake-2.8.12.1.tar.gz ")
os.system("cd repos && tar -xzf cmake*.tar.gz && cd cmake*/ && ./configure && make all")
os.system("cd repos && wget \"http://downloads.sourceforge.net/project/dclib/dlib/v18.5/dlib-18.5.tar.bz2?r=http%3A%2F%2Fdlib.net%2Fcompile.html&ts=1390320796&use_mirror=hivelocity\"")
action="cd repos && mv dlib* dlib.tar.bz2 "\
      +"&& bunzip2 dlib.tar.bz2 "\
      +"&& tar -xf dlib.tar "\
      +"&& mv dlib-18.5 dlib"
os.system(action)

os.system("cd repos && wget https://github.com/chokkan/liblbfgs/archive/master.zip ")
action="cd repos && mv master.zip liblbfgs.zip && unzip liblbfgs.zip "\
       "&& cd liblbfgs* && ./autogen.sh && ./configure  --enable-sse2 "\
       "&& make all "
