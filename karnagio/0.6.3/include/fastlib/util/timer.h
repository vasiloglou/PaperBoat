/*
Copyright © 2010, Ismion Inc
All rights reserved.
http://www.ismion.com/

Redistribution and use in source and binary forms, with or without
modification IS NOT permitted without specific prior written
permission. Further, neither the name of the company, Ismion
Inc, nor the names of its employees may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE Ismion Inc "AS IS" AND ANY
EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COMPANY BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef FL_LITE_FASTLIB_UTIL_TIMER_H
#define FL_LITE_FASTLIB_UTIL_TIMER_H

#ifndef WIN32
#include <sys/time.h>
#endif


namespace fl {
namespace util {
// We have 2 completely different class definitions because
// the timer utility used in linux is more accurate and platform
// processor independant. Windows implementation is accurate upto
// seconds only.
#ifdef WIN32
class Timer {

  public:

    void Start() {
      time(&start_);
    }

    void End() {
      time(&end_);
    }

    void Reset() {
      checkpoints_.clear();
    }

    /**
    * This is like lap time in a race. Can be used to store intermediate
    * timing results. Returns the id of the check point which can later
    * be used to get the elapsed time for this check point. Id's start
    * at 0;
    *
    */
    int CheckPoint() {
      time_t curr_time;
	  time(&curr_time);
      checkpoints_.push_back(curr_time);
      return (checkpoints_.size() - 1);
    }

    std::string GetTotalElapsedTimeString() {
		char val[100];
		sprintf(val, "%g", GetTotalElapsedTime());
		return "" + std::string(val);
    }

    double GetTotalElapsedTime() {
      return difftime(end_, start_);
    }

    /**
     * Given a checkpoint id it returns the amount of time
     * elapsed in seconds from the time start was called
     * till the checkpoint was initiated.
     */
    double GetElapsedTime(int checkpoint_id) {
      return difftime(checkpoints_[checkpoint_id], start_);
    }

  private:
    time_t start_, end_;
    std::vector<time_t> checkpoints_;
};


#else
/**
 * This is a timer class for benchmarking programs.
 */
class Timer {

  public:

    void Start() {
      gettimeofday(&start_, NULL);
    }

    void End() {
      gettimeofday(&end_, NULL);
    }

    void Reset() {
      checkpoints_.clear();
    }

    /**
    * This is like lap time in a race. Can be used to store intermediate
    * timing results. Returns the id of the check point which can later
    * be used to get the elapsed time for this check point. Id's start
    * at 0;
    *
    */
    int CheckPoint() {
      timeval curr_time;
      gettimeofday(&curr_time, NULL);
      checkpoints_.push_back(curr_time);
      return (checkpoints_.size() - 1);
    }

    std::string GetTotalElapsedTimeString() {
      timeval result;
      timersub(&end_, &start_, &result);
      char str[30]; // big enough
      sprintf(str, "%.6f", (result.tv_sec + result.tv_usec / 1000000.0));
      return std::string(str);
    }

    double GetTotalElapsedTime() {
      timeval result;
      timersub(&end_, &start_, &result);
      return (result.tv_sec + result.tv_usec / 1000000.0);
    }

    /**
     * Given a checkpoint id it returns the amount of time
     * elapsed in seconds from the time start was called
     * till the checkpoint was initiated.
     */
    double GetElapsedTime(int checkpoint_id) {
      //DEBUG_BOUNDS(checkpoint_id, checkpoints_.size());
      timeval result;
      timersub(&checkpoints_[checkpoint_id], &start_, &result);
      return (result.tv_sec + result.tv_usec / 1000000.0);
    }

  private:
    timeval start_, end_;
    std::vector<timeval> checkpoints_;
};
#endif

}
}

#endif
