// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>
#include <set>
#include <mutex>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"

using namespace std;

/* Editing is Prohibited*/
typedef float ScoreT;
const float kDamp = 0.85;
vector<double> total_proc_times;

vector<pair<ScoreT, NodeID>> PrintTopScores(const Graph  &g, ScoreT *scores);
bool CompareScores(vector<pair<ScoreT, NodeID>> &result, vector<pair<ScoreT, NodeID>> &answer);
/* Editing is Prohibited */

// PageRank Reference function
void PageRank(const Graph &g, ScoreT *scores, int num_iterations) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  const ScoreT init_score = 1.0f / g.num_nodes();
  for (NodeID u=0; u < g.num_nodes(); u++) {
    scores[u] = init_score;
  }
  for (int iter=0; iter < num_iterations; iter++) {
    pvector<ScoreT> incoming_scores(g.num_nodes(), 0.0f);    

    #pragma omp parallel for
    for (NodeID u=0; u < g.num_nodes(); u++) {
      ScoreT sum = 0.0f;
      for (NodeID v: g.in_neigh(u)){
        sum += scores[v]/ g.out_degree(v);
      }
      incoming_scores[u] = sum;
    }
    // #pragma omp parallel for 
    // for (NodeID u=0; u<g.num_nodes(); u++){
    //   for (NodeID v : g.out_neigh(u))
    //   {
    //     incoming_scores[v] += scores[u] / g.out_degree(u);
    //   }
    // }

    for (NodeID u=0; u < g.num_nodes(); u++){
      scores[u] = base_score + kDamp * incoming_scores[u];
    }
  }

  // printf("end iterations\n");
  return;
}



vector<pair<ScoreT, NodeID>> PageRank(const Graph &g, int num_iterations, int mode) {
  /* Editing is Prohibited*/
  Timer alloc_timer;
  alloc_timer.Start();
  double total_proc_time = 0;
  ScoreT* results = new ScoreT[g.num_nodes()]; // you must store pagerank result in here!
  ScoreT* contrib = new ScoreT[g.num_nodes()]; 
  /* Editing is Prohibited*/

  /* PREPROCESSING SECTION START */
  // TO-DO
  // preprocessing includes the following: data structure building (or modification), schedule planning (if any)
  // If you are not certain about putting something into this preprocessing section, please ask TAs.








  /* PREPROCESSING SECTION END   */

  /* Editing is Prohibited*/
  alloc_timer.Stop();
  PrintTime("Preprocessing Time ", alloc_timer.Seconds());

  Timer trial_timer;
  trial_timer.Start();
  /* Editing is Prohibited*/
 
  /* PAGERANK SECTION START */
  // TO-DO
  // You must store pagerank store to 'results' array by node_id ascending order



  PageRank(g, results, num_iterations);



  /* PAGERANK SECTION END   */

  /* Editing is Prohibited*/
  trial_timer.Stop();
  PrintTime("trial Time", trial_timer.Seconds());
  total_proc_time += trial_timer.Seconds();
  
  total_proc_times.push_back(total_proc_time);
  return PrintTopScores(g, results);
  /* Editing is Prohibited*/
}




/* WARNING!!!!*/
/* Don't touch below code!!!*/

vector<pair<ScoreT, NodeID>> PrintTopScores(const Graph &g, ScoreT *scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  k = min(k, static_cast<int>(top_k.size()));
  cout<<"Printing Top5 Ranks"<<endl;
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;

  return top_k;
}


bool CompareScores(vector<pair<ScoreT, NodeID>> &result, vector<pair<ScoreT, NodeID>> &answer) {
  int total_pass = 0;

  for (int i = 0; i < 5; i++){
    printf(" pp answer[%d]: %d, %f, res[%d] = %d, %f\n", 
        i, answer[i].first, answer[i].second, 
        i, result[i].first, result[i].second);
    bool check_1 = false, check_2 = false;
    if (result[i].second == answer[i].second)
      check_1 = true;

    if ((result[i].first / answer[i].first) >= 0.9 && (result[i].first / answer[i].first) <= 1.1)
      check_2 = true;

    if (check_1 && check_2)
      total_pass++;
  }

  if (total_pass == 5){
    cout<<"PASS!! your total pass: "<<total_pass<<endl;
    return true;
  }
  else {
    cout<<"NON-PASS!! your total pass: "<<total_pass<<endl;
    return false;
  }
}

int main(int argc, char* argv[]) {
  int total_pass = 0;

  CLIterApp cli(argc, argv, "pagerank", 100);
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  bool answer_exist = (cli.answer_file_name() != "");
  vector<pair<ScoreT, NodeID>> answer;
  vector<pair<ScoreT, NodeID>> result;

  if (answer_exist)
    answer = b.ReadAnswerFile();

  for(int t=0;t<cli.num_trials();t++) {
    result = PageRank(g, cli.num_iters(), cli.mode());
    if (answer_exist) {
      if (CompareScores(result, answer))
        total_pass++;
    }
  }

  if (answer_exist)
    cout<<"PageRank End. Your Pass Score: "<<total_pass<<", Mininum Runtime: "<<*min_element(total_proc_times.begin(), total_proc_times.end())<<endl;
  else
    cout<<"PageRank End. Minimum Runtime: "<<*min_element(total_proc_times.begin(), total_proc_times.end())<<endl;

  return 0;
}
