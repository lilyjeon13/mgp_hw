#include "smith_waterman_parallel.h"
//#include <algorithm>
#include "utils.h"
#include <stdio.h>

using namespace Algorithms;

SmithWatermanParallel::SmithWatermanParallel(int seq1Length, int seq2Length, char* seq1, char* seq2, int gapOp, int gapEx):SimilarityAlgorithmParallel(seq1Length, seq2Length), gapOp(gapOp), gapEx(gapEx)
{
  A = new int*[seq1Length + 1];
  E = new int*[seq1Length + 1]; //left matrix
  F = new int*[seq1Length + 1]; //up matrix
  B = new BackUpStruct*[seq1Length + 1];

  A[0] = new int[(seq1Length + 1) * (seq2Length + 1)](); //intializ to 0
  E[0] = new int[(seq1Length + 1) * (seq2Length + 1)]();
  F[0] = new int[(seq1Length + 1) * (seq2Length + 1)]();
  B[0] = new BackUpStruct[(seq1Length + 1) * (seq2Length + 1)]();

  for (int i = 1; i < seq1Length + 1; i++)
  {
    A[i] = A[0] + (seq2Length + 1)*i;
    E[i] = E[0] + (seq2Length + 1)*i;
    F[i] = F[0] + (seq2Length + 1)*i;
    B[i] = B[0] + (seq2Length + 1)*i;
  }

  setSeq1(seq1, seq1Length);
  setSeq2(seq2, seq2Length);
}

int SmithWatermanParallel::matchMissmatchScore(char a, char b) {
  if (a == b)
    return matchScore;
  else
    return missmatchScore;
}  /* End of matchMissmatchScore */


void SmithWatermanParallel::FillCell(int i, int j)
{

      //printf("at %d, %d = %c %c\n", i, j, seq1[i-1], seq2[j-1]);
      E[i][j] = MAX(E[i][j - 1] - gapEx, A[i][j - 1] - gapOp);
      B[i][j - 1].continueLeft = (E[i][j] == E[i][j - 1] - gapEx);
      F[i][j] = MAX(F[i - 1][j] - gapEx, A[i - 1][j] - gapOp);
      B[i - 1][j].continueUp = (F[i][j] == F[i - 1][j] - gapEx);

      A[i][j] = MAX3(E[i][j], F[i][j], A[i - 1][j - 1] + matchMissmatchScore(seq1[i-1], seq2[j-1]));
      A[i][j] = MAX(A[i][j], 0);


      if (A[i][j] == 0)
        B[i][j].backDirection = stop; //SPECYFIC FOR SMITH WATERMAN
      else if(A[i][j] == (A[i - 1][j - 1] + matchMissmatchScore(seq1[i-1], seq2[j-1])))
        B[i][j].backDirection = crosswise;
      else if(A[i][j] == E[i][j])
        B[i][j].backDirection = left;
      else //if(A[i][j] == F[i][j])
        B[i][j].backDirection = up;


      if(A[i][j] > maxVal)
      {
        maxX = j;
        maxY = i;
        maxVal = A[i][j];
      }

    }
void SmithWatermanParallel::FillMatrices()
{
  /*
   *   s e q 2
   * s
   * e
   * q
   * 1
   */
  //E - responsible for left direction
  //F - responsible for up   direction

  maxVal = INT_MIN;

  int minSeq = std::min(seq1Length, seq2Length);
  int maxSeq = std::max(seq1Length, seq2Length);
  for (int sum = 2; sum <= seq1Length + seq2Length; sum ++){
    if (minSeq+1 >= sum){
      #pragma omp parallel for 
      for(int i=1; i<sum; i++){
        FillCell(i, sum-i);
      }
    }else{
      #pragma omp parallel for 
      for (int i=minSeq; i>=std::max(1, sum-maxSeq); i--){
        if (minSeq == seq1Length){
          FillCell(i, sum-i);
        }else{
          FillCell(sum-i, i);
        }
      }
    }
    if (minSeq+1 < sum){
      #pragma omp parallel for 
      for(int i=1; i<=minSeq; i++){
        if (minSeq == seq1Length){
          FillCell(i, sum-i);
        }else{
          FillCell(sum-i, i);
        }
      }
    }
  }

  // for (int i = 1; i <= seq1Length; i++)
  // {
  //   for (int j = 1; j <= seq2Length; j++)
  //   {
  //     FillCell(i, j);
  //   }
    
  // }
  printf("Matrix Filled: maxY %d maxX %d maxVal %d\n", maxY, maxX, maxVal);
}

void SmithWatermanParallel::BackwardMoving()
{
  //BACKWARD MOVING
  int carret = 0;

  int y = maxY;
  int x = maxX;

  BackDirection prev = crosswise;
  while(B[y][x].backDirection != stop)
  {
    path.push_back(std::make_pair(y, x));
    if (prev == up && B[y][x].continueUp) //CONTINUE GOING UP
    {                                          //GAP EXTENSION
      carret++;
      y--;
    }
    else if (prev == left && B[y][x].continueLeft) //CONTINUE GOING LEFT
    {                                         //GAP EXTENSION
      carret++;
      x--;
    }
    else
    {
      prev = B[y][x].backDirection;
      if(prev == up)
      {
        carret++;
        y--;
      }
      else if(prev == left)
      {
        carret++;
        x--;
      }
      else //prev == crosswise
      {
        carret++;
        x--;
        y--;
      }
    }
  }
  printf("Backward Moving: destY %d destX %d\n", y, x);
}
