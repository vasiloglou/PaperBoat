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
#ifndef FL_LITE_FASTLIB_TABLE_CLIENT_TABLE_H_
#define FL_LITE_FASTLIB_TABLE_CLIENT_TABLE_H_
#include <vector>
#include <set>  
#include "table.h"
#include "boost/mpi/communicator.hpp"
namespace fl {
namespace table {

  /**
   * @brief The ClientTable inherits from the Normal Table
   *        It has a vector of mpi::communicators (worlds)
   *        In general it is connected to N master tables 
   *        Different sets of points report to different master
   *        tables. Master tables have not common points. Client
   *        tables can have common points. 
   */  
  template<typename TemplateMap>
  class ClientTable : public Table<TemplateMap> {
    public:
      typedef typename fl::table::Table<TemplateMap>::Point_t Point_t;
      ClientTable(boost::mpi::communicator &world,
                  std::vector<index_t> &point2world);

      /**
       * @brief This function will push a point update to the master.
       *        It sends asynchronously the point to a master table
       */
      void PushToMaster(index_t point_id);
      /** @brief Here is the assumption here
       *         The Master table will send updates whenever
       *         it thinks it is appropriate
       *         PullFromMaster function just look if there is any message
       *         for them and pick it. We pull whatever is available
       */
      void PullFromMaster(index_t point_id);
  
    private:
      std::vector<index_t> point2world_;
      std::vector<boost::mpi::communicator> worlds_;
      std::vector<boost::mpi::request> requests_;
  
  };
  
  template<typename TemplateMap>
  ClientTable<TemplateMap>::ClientTable(boost::mpi::communicator &world,
                                        std::vector<index_t> &point2world) {
    std::set<index_t> unique_worlds;
    for(index_t i=0; i<point2world.size(); ++i) {
      unique_worlds.insert(point2world[i]);
    }
    worlds_.resize(unique_worlds.size());
    std::set<index_t>::iterator it;
    index_t i=0;
    for(it=unique_worlds.begin(); it!=unique_worlds.end(); ++it) {
      worlds_[i] = world.split(*it);
      ++i;
    }
    point2world_=point2world;
    requests_.resize(point2world.size());
  }

  template<typename TemplateMap>
  void ClientTable<TemplateMap>::PushToMaster(index_t point_id) {
    // check to see if the old request has been received. 
    // If not just cancel it
    if (!requests_[point_id].test()) {
      requests_[point_id].cancel();
    }
    Point_t point;
    this->get(point_id, &point);
    requests_[point_id] = worlds_[point2world_[point_id]].isend(0, point_id, point);
  }
  
  template<typename TemplateMap>
  void ClientTable<TemplateMap>::PullFromMaster(index_t point_id) {
    Point_t point;
    this->get(point_id, &point);
    boost::optional<boost::mpi::status> msg=
      worlds_[point2world_[point_id]].iprobe(0, point_id);
    if (!msg==false) {
      worlds_[point2world_[point_id]].recv(0, point_id, point);  
    }
  }


}}


#endif

