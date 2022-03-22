# Homework 2

## Summary

You should improve a performance of locked open addressing hash table.

Go to `better_locked_hash_table.h` and implement `TODO`

```
$ make remote_better
$ cat better.out
TABLE_SIZE 10000000 init: 4000000 new: 4000000 NT: 16 additional_reads: 9 use_custom: 1
user-defined HT1 1
start filling
init hash table took 1.6869 sec
start test
test 36000000 ops took @@@@@@@ sec
sanity check PASSED: 
```

## Functions

```
# Run locally
make
make run
# Compile
make HTtest
# Clean
make clean
# Submit base job to condor
make remote_base
# Submit better job to condor
make remote_better
# Check queue of condor
make queue
# Check status of condor
make status
# Remove jobs from condor
make remove
```

## References

- [std::thread - cplusplus.com](http://www.cplusplus.com/reference/thread/thread/)
- [HTCondor commands cheat-sheet](https://raggleton.github.io/condor-cheatsheet/)
