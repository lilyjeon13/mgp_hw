#ifndef _BETTER_LOCKED_HASH_TABLE_H_
#define _BETTER_LOCKED_HASH_TABLE_H_


#include <iostream>
#include <mutex>
#include <thread>
#include "hash_table.h"
#include "bucket.h"
#include <cmath>

class better_locked_probing_hash_table : public hash_table {

  private:
    Bucket* table;
    const int TABLE_SIZE; //we do not consider resizing. Thus the table has to be larger than the max num items.
    std::mutex global_mutex; 

    /* TODO: put your own code here  (if you need something)*/
    /****************/
    std::mutex* locks; // usually 16 locks
    const int LOCK_SIZE;

    virtual uint64_t get_lock_index(uint64_t index){
      return index % LOCK_SIZE;
    }

    /****************/
    /* TODO: put your own code here */

    public:

    better_locked_probing_hash_table(int table_size):TABLE_SIZE(table_size), LOCK_SIZE(table_size/16){
      this->table = new Bucket[TABLE_SIZE]();
      for(int i=0;i<TABLE_SIZE;i++) {
        this->table[i].valid=0; //means empty
      }
      this->locks = new std::mutex[LOCK_SIZE]();
    }

    virtual uint32_t hash(uint32_t x)  
    {
      //https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      x = (x >> 16) ^ x;
      return (x % TABLE_SIZE); 
    }

    virtual uint32_t hash_next(uint32_t key, uint32_t prev_index)
    {
      // //linear probing. no special secondary hashfunction
      return ((prev_index + 1)% TABLE_SIZE); 
    }
    virtual uint32_t hash_quad_next(uint32_t key, uint32_t origin_index, uint32_t iter_count)
    {
      //quadratic probing
      return ((origin_index + int(pow(iter_count,2)))% TABLE_SIZE);
    }

    //the buffer has to be allocated by the caller
    bool read(uint32_t key, uint64_t* value_buffer){
      /* TODO: put your own read function here */
      /****************/
      uint64_t index = this->hash(key);
      int probe_count=0;
      uint64_t lock_index = get_lock_index(index);
      uint32_t iter_count = 0;
      uint32_t origin_index = index;

      locks[lock_index].lock();
      while(table[index].valid == true) {
        if(table[index].key == key) {
          *value_buffer = table[index].value;
          locks[lock_index].unlock();
          return true;
        } else {
          probe_count++;
          iter_count++;
          locks[lock_index].unlock();
          index = this->hash_quad_next(key, origin_index, iter_count);
          if(probe_count >= TABLE_SIZE) break;
          lock_index = get_lock_index(index);
          locks[lock_index].lock();
        }
      }//end while

      //If you reached here, you either encountered an empty slot or the table is full. In any case, the item you're looking for is not here 
      return false;

      /****************/
      /* TODO: put your own read function here */
    }


    bool insert(uint32_t key, uint64_t value) {
      /* TODO: put your own insert function here */
      /****************/
      // lock guard
      // std::lock_guard<std::mutex> lock(global_mutex);
      uint64_t index = this->hash(key);
      int probe_count=0;
      uint64_t lock_index = get_lock_index(index);
      uint32_t iter_count = 0;
      uint32_t origin_index = index;

      locks[lock_index].lock();
      while(table[index].valid == true) {
        if(table[index].key == key) {
          //found it already there. just modify
          break;
        } else {
          probe_count++;
          iter_count++;
          locks[lock_index].unlock();
          index = this->hash_quad_next(key, origin_index, iter_count);
          if(probe_count >= TABLE_SIZE) return false; //could not add because the table was full
          lock_index = get_lock_index(index);
          locks[lock_index].lock();
        }
      }//end while

      //if you came here, 
      //1. You found a bucket with the same key
      //2. You encountered an empty bucket

      //You might be overwriting in case 1, but it is still functionally correct
      table[index].valid = true; 
      table[index].key   = key; 
      table[index].value = value;
      locks[lock_index].unlock();
      return true;

      /****************/
      /* TODO: put your own insert function here */
    }

    int num_items() {
      int count=0;
      for(int i=0;i<TABLE_SIZE;i++) {
        if(table[i].valid==true) count++;
      }
      return count;
    }
};

#endif


