#./bin/bash
# build_and_run name_of_cpp_file M N n_procs

if [ $# -le 3 ]
  then
    echo "Some argument is missing!"
    exit 1
fi

if [ $# -eq 4 ]
  then
    mpicxx -o ik_build/${1}_hands "${1}".cpp -std=c++11 -Wall -Wpedantic -Wconversion -lm && mpirun -np ${4} ./ik_build/${1}_hands $2 $3
  else
    if [ $# -eq 5 ]
      then
        export OMP_NUM_THREADS=4
        mpicxx -o ik_build/${1}_hands_omp "${1}".cpp -std=c++11 -Wall -Wpedantic -Wconversion -fopenmp -lm && mpiexec -np ${4} ./ik_build/${1}_hands_omp $2 $3
    else
      echo "DEBUG enabled"
      # MPI leaks
      mpicxx -o ik_build/${1}_hands "${1}".cpp -std=c++11 -Wall -Wpedantic -Wconversion -g -lm && mpirun -np ${4} valgrind --leak-check=full ./ik_build/${1}_hands $2 $3
    fi
fi

# for gdb: mpirun -np <NP> xterm -e gdb ./program