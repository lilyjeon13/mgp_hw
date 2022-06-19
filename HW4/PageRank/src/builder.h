// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef BUILDER_H_
#define BUILDER_H_

#include <algorithm>
#include <cinttypes>
#include <fstream>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include "command_line.h"
#include "platform_atomics.h"
#include "graph.h"
#include "print_util.h"
#include "pvector.h"
#include "reader.h"
#include "timer.h"


/*
GAP Benchmark Suite
Class:  BuilderBase
Author: Scott Beamer

Given arguements from the command line (cli), returns a built graph
 - MakeGraph() will parse cli and obtain edgelist and call
   MakeGraphFromEL(edgelist) to perform actual graph construction
 - edgelist can be from file (reader) or synthetically generated (generator)
 - Common case: BuilderBase typedef'd (w/ params) to be Builder (benchmark.h)
*/


template <typename NodeID_, typename DestID_ = NodeID_,
          bool invert = true>
class BuilderBase {
  typedef EdgePair<NodeID_, DestID_> Edge;
  typedef pvector<Edge> EdgeList;

  const CLBase &cli_;
  int64_t num_nodes_ = -1;

 public:
  explicit BuilderBase(const CLBase &cli) : cli_(cli) {
  }

  DestID_ GetSource(EdgePair<NodeID_, NodeID_> e) {
    return e.u;
  }

  NodeID_ FindMaxNodeID(const EdgeList &el) {
    NodeID_ max_seen = 0;
    #pragma omp parallel for reduction(max : max_seen)
    for (auto it = el.begin(); it < el.end(); it++) {
      Edge e = *it;
      max_seen = std::max(max_seen, e.u);
      max_seen = std::max(max_seen, (NodeID_) e.v);
    }
    return max_seen;
  }

  pvector<NodeID_> CountDegrees(const EdgeList &el, bool transpose) {
    pvector<NodeID_> degrees(num_nodes_, 0);
    #pragma omp parallel for
    for (auto it = el.begin(); it < el.end(); it++) {
      Edge e = *it;
      if (!transpose)
        fetch_and_add(degrees[e.u], 1);
      if (transpose)
        fetch_and_add(degrees[(NodeID_) e.v], 1);
    }
    return degrees;
  }

  static
  pvector<SGOffset> PrefixSum(const pvector<NodeID_> &degrees) {
    pvector<SGOffset> sums(degrees.size() + 1);
    SGOffset total = 0;
    for (size_t n=0; n < degrees.size(); n++) {
      sums[n] = total;
      total += degrees[n];
    }
    sums[degrees.size()] = total;
    return sums;
  }

  static
  pvector<SGOffset> ParallelPrefixSum(const pvector<NodeID_> &degrees) {
    const size_t block_size = 1<<20;
    const size_t num_blocks = (degrees.size() + block_size - 1) / block_size;
    pvector<SGOffset> local_sums(num_blocks);
    #pragma omp parallel for
    for (size_t block=0; block < num_blocks; block++) {
      SGOffset lsum = 0;
      size_t block_end = std::min((block + 1) * block_size, degrees.size());
      for (size_t i=block * block_size; i < block_end; i++)
        lsum += degrees[i];
      local_sums[block] = lsum;
    }
    pvector<SGOffset> bulk_prefix(num_blocks+1);
    SGOffset total = 0;
    for (size_t block=0; block < num_blocks; block++) {
      bulk_prefix[block] = total;
      total += local_sums[block];
    }
    bulk_prefix[num_blocks] = total;
    pvector<SGOffset> prefix(degrees.size() + 1);
    #pragma omp parallel for
    for (size_t block=0; block < num_blocks; block++) {
      SGOffset local_total = bulk_prefix[block];
      size_t block_end = std::min((block + 1) * block_size, degrees.size());
      for (size_t i=block * block_size; i < block_end; i++) {
        prefix[i] = local_total;
        local_total += degrees[i];
      }
    }
    prefix[degrees.size()] = bulk_prefix[num_blocks];
    return prefix;
  }


    void MakeCSR(const EdgeList &el, bool transpose, 
      index_t** p_vertex_table, DestID_** p_edge_table ) {
    pvector<NodeID_> degrees = CountDegrees(el, transpose);
    pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
    std::cout<<"alloc edgelist "<<offsets[num_nodes_]<<std::endl;
    *p_edge_table = new DestID_[offsets[num_nodes_]];
    *p_vertex_table = CSRGraph<NodeID_, DestID_>::GenVertexTable(offsets, *p_edge_table);
    #pragma omp parallel for
    for (auto it = el.begin(); it < el.end(); it++) {
      Edge e = *it;

      if (!transpose)
        (*p_edge_table)[fetch_and_add(offsets[e.u], 1)] = e.v;
      if (transpose)
        (*p_edge_table)[fetch_and_add(offsets[static_cast<NodeID_>(e.v)], 1)] =
            GetSource(e);
    }
  }

  CSRGraph<NodeID_, DestID_, invert> MakeGraphFromEL(EdgeList &el) {
    index_t* vertex_table, *inv_vertex_table;
    DestID_* edge_table, *inv_edge_table;

    Timer t;
    t.Start();
    if (num_nodes_ == -1)
      num_nodes_ = FindMaxNodeID(el)+1;

    std::cout<<"EL size : "<<el.size()<<std::endl;
    MakeCSR(el, false, &vertex_table, &edge_table);
    MakeCSR(el, true, &inv_vertex_table, &inv_edge_table);
    t.Stop();
    PrintTime("Build Time", t.Seconds());
    return CSRGraph<NodeID_, DestID_, invert>(num_nodes_, vertex_table, edge_table,
        inv_vertex_table, inv_edge_table);
  }

  CSRGraph<NodeID_, DestID_, invert> MakeGraph() {
    CSRGraph<NodeID_, DestID_, invert> g;
    {  // extra scope to trigger earlier deletion of el (save memory)
      EdgeList el;
      Reader<NodeID_, DestID_, invert> r(cli_.filename());
      el = r.ReadFile();
      g = MakeGraphFromEL(el);
    }
    return g;
  }

  std::vector<std::pair<float, NodeID_>> ReadAnswerFile() {
    std::vector<std::pair<float, NodeID_>> ret;
    std::ifstream afile(cli_.answer_file_name());
    NodeID_ id;
    float score;

    if (!afile.is_open()) {
      std::cout<< "Couldn't open answer file " << cli_.answer_file_name() << std::endl;
      std::exit(-1);
    }

    for (int i = 0; i < 5; i++) {
      afile >> score >> id;
      ret.push_back(std::make_pair(score, id));
    }

    return ret;
  }
};

#endif  // BUILDER_H_
