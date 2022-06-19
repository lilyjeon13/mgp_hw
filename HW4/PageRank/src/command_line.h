// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef COMMAND_LINE_H_
#define COMMAND_LINE_H_

#include <getopt.h>

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <string>
#include <vector>


/*
GAP Benchmark Suite
Class:  CLBase
Author: Scott Beamer

Handles command line argument parsing
 - Through inheritance, can add more options to object
 - For example, most kernels will use CLApp
*/


class CLBase {
 protected:
  int argc_;
  char** argv_;
  std::string name_;
  std::string get_args_ = "f:g:m:hsu:c:";
  std::vector<std::string> help_strings_;

  int mode_;
  int scale_ = -1;
  std::string filename_ = "";
  std::string answer_file_name_ = "";
  bool symmetrize_ = false;
  bool uniform_ = false;

  void AddHelpLine(char opt, std::string opt_arg, std::string text,
                   std::string def = "") {
    const int kBufLen = 100;
    char buf[kBufLen];
    if (opt_arg != "")
      opt_arg = "<" + opt_arg + ">";
    if (def != "")
      def = "[" + def + "]";
    sprintf(buf, " -%c %-9s: %-57s%7s", opt, opt_arg.c_str(),
            text.c_str(), def.c_str());
    help_strings_.push_back(buf);
  }

 public:
  CLBase(int argc, char** argv, std::string name = "") :
         argc_(argc), argv_(argv), name_(name) {
    AddHelpLine('h', "", "print this help message");
    AddHelpLine('f', "file", "load graph from file");
    AddHelpLine('c', "answer_file", "load answer file would be compared");
    AddHelpLine('m', "mode", "which custom mode of application");
  }

  bool ParseArgs() {
    signed char c_opt;
    extern char *optarg;          // from and for getopt
    while ((c_opt = getopt(argc_, argv_, get_args_.c_str())) != -1) {
      HandleArg(c_opt, optarg);
    }
    if ((filename_ == "") && (scale_ == -1)) {
      std::cout << "No graph input specified. (Use -h for help)" << std::endl;
      return false;
    }
    if (scale_ != -1)
      symmetrize_ = true;
    return true;
  }

  void virtual HandleArg(signed char opt, char* opt_arg) {
    switch (opt) {
      case 'f': filename_ = std::string(opt_arg);           break;
      case 'h': PrintUsage();                               break;
      case 'c': answer_file_name_ = std::string(opt_arg);   break;
      case 'm': mode_ = atoi(opt_arg);   break;
    }
  }

  void PrintUsage() {
    std::cout << name_ << std::endl;
    // std::sort(help_strings_.begin(), help_strings_.end());
    for (std::string h : help_strings_)
      std::cout << h << std::endl;
    std::exit(0);
  }

  int mode() const {return mode_;}
  std::string filename() const { return filename_; }
  std::string answer_file_name() const { return answer_file_name_; }
};



class CLApp : public CLBase {
  int num_trials_ = 10;

 public:
  CLApp(int argc, char** argv, std::string name) : CLBase(argc, argv, name) {
    get_args_ += "an:r:";
    char buf[30];
    sprintf(buf, "%d", num_trials_);
    AddHelpLine('n', "n", "perform n trials", buf);
  }

  void HandleArg(signed char opt, char* opt_arg) override {
    switch (opt) {
      case 'n': num_trials_ = atoi(opt_arg);            break;
      default: CLBase::HandleArg(opt, opt_arg);
    }
  }

  int num_trials() const { return num_trials_; }
};



class CLIterApp : public CLApp {
  int num_iters_;

 public:
  CLIterApp(int argc, char** argv, std::string name, int num_iters) :
    CLApp(argc, argv, name), num_iters_(num_iters) {
    get_args_ += "k:";
    char buf[15];
    sprintf(buf, "%d", num_iters_);
    AddHelpLine('k', "k", "perform k iterations", buf);
  }

  void HandleArg(signed char opt, char* opt_arg) override {
    switch (opt) {
      case 'k': num_iters_ = atoi(opt_arg);            break;
      default: CLApp::HandleArg(opt, opt_arg);
    }
  }

  int num_iters() const { return num_iters_; }
};

#endif  // COMMAND_LINE_H_
