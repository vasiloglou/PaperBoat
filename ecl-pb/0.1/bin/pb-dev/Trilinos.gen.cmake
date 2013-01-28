find_library(TriUtils_LIBRARY NAMES trilinos_triutils triutils)
find_library(Teuchos_LIBRARY NAMES trilinos_teuchos teuchos)
find_library(Epetra_LIBRARY NAMES trilinos_epetra epetra)
find_library(EpetraExt_LIBRARY NAMES trilinos_epetraext epetraext)
find_library(Ifpack_LIBRARY NAMES trilinos_ifpack ifpack)
find_library(AztecOO_LIBRARY NAMES trilinos_aztecoo aztecoo)
find_library(Belos_LIBRARY NAMES trilinos_belos belos)
find_library(Anasazi_LIBRARY NAMES trilinos_anasazi anasazi)
find_library(Amesos_LIBRARY NAMES trilinos_amesos amesos)

find_path(Trilinos_INCLUDE_DIR Amesos.h PATH_SUFFIXES trilinos)
if (Trilinos_INCLUDE_DIR)
  set(Trilinos_LIBRARIES
    ${TriUtils_LIBRARY}
    ${Teuchos_LIBRARY}
    ${Epetra_LIBRARY}
    ${EpetraExt_LIBRARY}
    ${Ifpack_LIBRARY}
    ${AztecOO_LIBRARY}
    ${Belos_LIBRARY}
    ${Anasazi_LIBRARY}
    ${Amesos_LIBRARY})
  include_directories(${Trilinos_INCLUDE_DIR})
  
  #find_package(EXPAT)
  include(FindEXPAT)
  # set(EXPAT_LIBRARIES /usr/lib/i386-linux-gnu/libexpat.so /usr/lib/i386-linux-gnu/libdl.so)
  if(EXPAT_FOUND)
   message(STATUS ${EXPAT_LIBRARIES})
   list(APPEND Trilinos_LIBRARIES ${EXPAT_LIBRARIES})
   include_directories(${EXPAT_INCLUDE_DIRS})
 endif()
  
  
  
  set(MPI_COMPILER mpicxx.openmpi)
  
  find_package(MPI)
  
  if(MPI_FOUND)
    list(APPEND Trilinos_LIBRARIES ${MPI_LIBRARIES})
    include_directories(${MPI_INCLUDE_PATH})
    append(CMAKE_CXX_FLAGS " ${MPI_COMPILE_FLAGS}")
    append(CMAKE_EXE_LINKER_FLAGS " ${MPI_LINK_FLAGS}")
  endif()
  
  
  
  set(GenCMake_LIBRARIES ${Trilinos_LIBRARIES} ${GenCMake_LIBRARIES})
endif()  
