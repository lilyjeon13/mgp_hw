# Homework 1

## Summary

You should implement a simple filter on 1D array for parallelization

Go to `filter.cc` and implement `TODO`

```
$ make remote
$ cat out.txt
init took 0.878147 sec
serial 1D filter took 1.33711 sec
parallel 1D filter took 0.433189 sec
PASS
```

## Functions

```
# Run locally
make
make run
# Compile
make filter
# Clean
make clean
# Submit job to condor
make remote
# Check queue of condor
make queue
# Check status of condor
make status
# Remove jobs from condor
make remove
```

## References

- [std::thread - cplusplus.com](http://www.cplusplus.com/reference/thread/thread/)
- [OpenMP 4.5 C/C++ Syntax Guide](https://www.openmp.org/wp-content/uploads/OpenMP-4.5-1115-CPP-web.pdf)
- [HTCondor commands cheat-sheet](https://raggleton.github.io/condor-cheatsheet/)
- [CMake Cheatsheet](https://github.com/mortennobel/CMake-Cheatsheet/blob/master/CMake_Cheatsheet.pdf)
