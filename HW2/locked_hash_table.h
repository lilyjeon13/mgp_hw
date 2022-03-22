#ifndef _LOCKED_HASH_TABLE_H_
#define _LOCKED_HASH_TABLE_H_


#include <iostream>
#include <mutex>
#include <thread>
#include "hash_table.h"
#include "bucket.h"

class locked_probing_hash_table : public hash_table {

  private:
    Bucket* table;
    const int TABLE_SIZE; //we do not consider resizing. Thus the table has to be larger than the max num items.
    std::mutex global_mutex; 

    public:

    locked_probing_hash_table(int table_size):TABLE_SIZE(table_size){
      this->table = new Bucket[TABLE_SIZE]();
      for(int i=0;i<TABLE_SIZE;i++) {
        this->table[i].valid=0; //means empty
      }
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
      //linear probing. no special secondary hashfunction
      return ((prev_index + 1)% TABLE_SIZE); 
    }

    //the buffer has to be allocated by the caller
    bool read(uint32_t key, uint64_t* value_buffer){
      // Lock guard
      std::lock_guard<std::mutex> lock(global_mutex);
      uint64_t index = this->hash(key);
      int probe_count=0;

      while(table[index].valid == true) {
        if(table[index].key == key) {
          *value_buffer = table[index].value;
          return true;
        } else {
          probe_count++;
          index = this->hash_next(key, index);
          if(probe_count >= TABLE_SIZE) break;
        }
      }//end while

      //If you reached here, you either encountered an empty slot or the table is full. In any case, the item you're looking for is not here 
      return false;


    }


    bool insert(uint32_t key, uint64_t value) {
      // lock guard
      std::lock_guard<std::mutex> lock(global_mutex);
      uint64_t index = this->hash(key);
      int probe_count=0;

      while(table[index].valid == true) {
        if(table[index].key == key) {
          //found it already there. just modify
          break;
        } else {
          probe_count++;
          index = this->hash_next(key, index);
          if(probe_count >= TABLE_SIZE) return false; //could not add because the table was full
        }
      }//end while

      //if you came here, 
      //1. You found a bucket with the same key
      //2. You encountered an empty bucket

      //You might be overwriting in case 1, but it is still functionally correct
      table[index].valid = true; 
      table[index].key   = key; 
      table[index].value = value;
      return true;

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
