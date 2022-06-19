// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef GRAPH_H_
#define GRAPH_H_

#include <cinttypes>
#include <iostream>
#include <type_traits>
#include <set>
#include <assert.h>

#include "pvector.h"
#define DEBUG 0
#define debug_print(fmt, ...) \
              do { if (DEBUG) fprintf(stdout, fmt, __VA_ARGS__); } while (0)

/*
GAP Benchmark Suite
Class:  CSRGraph
Author: Scott Beamer

Simple container for graph in CSR format
 - Intended to be constructed by a Builder
 - To make weighted, set DestID_ template type to NodeWeight
 - MakeInverse parameter controls whether graph stores its inverse
*/

// Used to hold node & weight, with another node it makes a weighted edge

// Syntatic sugar for an edge
template <typename SrcT, typename DstT = SrcT>
struct EdgePair {
  SrcT u;
  DstT v;

  EdgePair() {}

  EdgePair(SrcT u, DstT v) : u(u), v(v) {}
};

// SG = serialized graph, these types are for writing graph to file
typedef int32_t SGID;
typedef EdgePair<SGID> SGEdge;
typedef int64_t SGOffset;

typedef unsigned int index_t;


template <class NodeID_, class DestID_ = NodeID_, bool MakeInverse = true>
class CSRGraph {
  // Used to access neighbors of vertex, basically sugar for iterators
  class Neighborhood {
    NodeID_ n_;
    index_t* vertex_table;
    DestID_* edge_table;
    public:
    Neighborhood(NodeID_ n, index_t* vertex_table, DestID_* edge_table) : n_(n), vertex_table(vertex_table), edge_table(edge_table){}
    typedef DestID_* iterator;
    iterator begin() 
    { 
      debug_print("begin n_ %d, g[n_] %d * %d\n",
        n_, vertex_table[n_], edge_table[vertex_table[n_]]); 
      return &edge_table[vertex_table[n_]]; 
    }
    
    iterator end()   
    { 
      debug_print("end n_ %d, g[n_+1] %d * %d\n",
        n_+1, vertex_table[n_+1], edge_table[vertex_table[n_+1]-1]);
      return &edge_table[vertex_table[n_+1]]; 
    }
  };

  void ReleaseResources() {
    if (out_vertex_table_ != nullptr)
      delete[] out_vertex_table_;
    if (out_edge_table_ != nullptr)
      delete[] out_edge_table_;
    if(directed_)
    {
      if (in_vertex_table_ != nullptr)
        delete[] in_vertex_table_;
      if (in_edge_table_ != nullptr)
        delete[] in_edge_table_;
    }
  }


 public:
  CSRGraph() : directed_(false), num_nodes_(-1), num_edges_(-1),
   out_vertex_table_(nullptr),in_vertex_table_(nullptr), 
   out_edge_table_(nullptr) , in_edge_table_(nullptr){}



  //Undirected edges
  CSRGraph(int64_t num_nodes, index_t* vertex_table, DestID_* edge_table ) :
    directed_(false), num_nodes_(num_nodes),
    out_vertex_table_(vertex_table), in_vertex_table_(vertex_table), 
    out_edge_table_(edge_table),     in_edge_table_(edge_table) {
      std::cout<<"undirected constructor"<<std::endl;
      num_edges_ = (vertex_table[num_nodes_] - vertex_table[0])/2;
    }

  //Directed graphs
  CSRGraph(int64_t num_nodes, 
      index_t* out_vertex_table,  DestID_* out_edge_table ,
      index_t* in_vertex_table, DestID_* in_edge_table ) :
    directed_(false), num_nodes_(num_nodes),
    out_vertex_table_(out_vertex_table), in_vertex_table_(in_vertex_table), 
    out_edge_table_(out_edge_table),     in_edge_table_(in_edge_table) {
      std::cout<<"directed constructor"<<std::endl;
      num_edges_ = (out_vertex_table_[num_nodes_] - out_vertex_table_[0]);
    }


  CSRGraph(CSRGraph&& other) : directed_(other.directed_),
  num_nodes_(other.num_nodes_), num_edges_(other.num_edges_),
  in_vertex_table_(other.in_vertex_table_),out_vertex_table_(other.out_vertex_table_),
  in_edge_table_(other.in_edge_table_) ,   out_edge_table_(other.out_edge_table_) {
    other.num_edges_ = -1;
    other.num_nodes_ = -1;
    other.in_vertex_table_= nullptr;
    other.out_vertex_table_= nullptr;
    other.in_edge_table_= nullptr;
    other.out_edge_table_= nullptr;
  }

  ~CSRGraph() {
    ReleaseResources();
  }

  CSRGraph& operator=(CSRGraph&& other) {
    if (this != &other) {
      ReleaseResources();
      directed_ = other.directed_;
      num_edges_ = other.num_edges_;
      num_nodes_ = other.num_nodes_;
      in_vertex_table_ = other.in_vertex_table_;
      in_edge_table_ =   other.in_edge_table_;
      out_vertex_table_ = other.out_vertex_table_;
      out_edge_table_ =   other.out_edge_table_;
      other.num_edges_ = -1;
      other.num_nodes_ = -1;
      other.in_vertex_table_= nullptr;
      other.in_edge_table_= nullptr;
      other.out_vertex_table_= nullptr;
      other.out_edge_table_= nullptr;
    }
    return *this;
  }

  bool directed() const {
    return directed_;
  }

  int64_t num_nodes() const {
    return num_nodes_;
  }

  int64_t num_edges() const {
    return num_edges_;
  }

  int64_t num_edges_directed() const {
    return directed_ ? num_edges_ : 2*num_edges_;
  }

  int64_t out_degree(NodeID_ v) const {
    return out_vertex_table_[v+1] - out_vertex_table_[v];
  }


  int64_t in_degree(NodeID_ v) const {
    return in_vertex_table_[v+1] - in_vertex_table_[v];
  }

  /*
  int64_t in_degree(NodeID_ v) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    debug_print("getting degree : u %d  in_indexv+1 %d  in_indexv %d res %d\n", v, in_index_[v+1], in_index_[v],in_index_[v+1] - in_index_[v]);
    return in_index_[v+1] - in_index_[v];
  }*/

  Neighborhood out_neigh(NodeID_ n) const {
    return Neighborhood(n, out_vertex_table_, out_edge_table_);
  }

  Neighborhood in_neigh(NodeID_ n) const {
    return Neighborhood(n, in_vertex_table_, in_edge_table_);
    /*debug_print("looking for in neighbor of %d\n", n);
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    return Neighborhood(n, in_index_);*/
  }


  void PrintStats() const {
    std::cout << "Graph has " << num_nodes_ << " nodes and "
              << num_edges_ << " ";
    if (!directed_)
      std::cout << "un";
    std::cout << "directed edges for degree: ";
    std::cout << num_edges_/num_nodes_ << std::endl;
  }

  void PrintTopology() const {
    for (NodeID_ i=0; i < num_nodes_; i++) {
      std::cout << i << ": ";
      for (DestID_ j : out_neigh(i)) {
        std::cout << j << " ";
      }
      std::cout << std::endl;
    }
  }

  
  static index_t* GenVertexTable(const pvector<SGOffset> &offsets, 
                                 DestID_* edge_table) {
    NodeID_ length = offsets.size();
    index_t* vertex_table = new index_t[length];
    #pragma omp parallel for
    for (NodeID_ n=0; n < length; n++)
      vertex_table[n] = offsets[n];
    return vertex_table;
  }

   public:
   //private:
  bool directed_;
  int64_t num_nodes_;
  int64_t num_edges_;

  //stores indices of edge_table
  index_t* out_vertex_table_;
  index_t* in_vertex_table_;

  //stores destination node ids
  DestID_* out_edge_table_;
  DestID_* in_edge_table_;
};

#endif  // GRAPH_H_
