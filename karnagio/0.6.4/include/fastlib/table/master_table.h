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
#ifndef FL_LITE_FASTLIB_TABLE_MASTER_TABLE_H_
#define FL_LITE_FASTLIB_TABLE_MASTER_TABLE_H_


#include <vector>
#include "fastlib/base/base.h"
#include "table.h"
#include "boost/mpi/communicator.hpp"

namespace fl {
namespace table {

  struct MasterTableArgs : public TableArgs {
    typedef boost::mpl::void_       AggregatorType;
    typedef TableArgs::DatasetType  DatasetType ;
    typedef TableArgs::SortPoints   SortPoints ;
  };

  /**
   * @brief The MasterTable inherits from the Normal Table
   *        It has a vector of mpi::communicators (worlds)
   *        In general it is connected to N master tables 
   *        Different sets of points report to different master
   *        tables. Master tables have not common points. Client
   *        tables can have common points. 
   */  
  template<typename TemplateMap>
  class MasterTable : public Table<TemplateMap> {
    public:
      typedef typename TemplateMap::AggregatorType Aggregator_t;
      typedef typename Table<TemplateMap>::Point_t Point_t;
      /**
       * @brief This is the constructor that needs a world
       *        and a color. For further initialization
       *        we use the traditional table Init functions
       *        
       */
      MasterTable(boost::mpi::communicator &world,
                  index_t world_color,
                  Aggregator_t* aggregator);
      /**
       * @brief Initializes all the parameters of the master table
       *        that have to do with communication
       *
       */
       void InitMasterParams();

      /**
       * @brief Pushes an update of a point to the slaves
       */
      void PushToClients(index_t point_id);
      /** @brief Here is the assumption here
       *         The Client table will send updates whenever
       *         it thinks it is appropriate
       *         PullRequest function just look if there is any message
       *         for them and pick it. We pull whatever is available
       */
      void PullFromClients(index_t point_id);

    private:
      boost::mpi::communicator world_;
      std::vector<boost::mpi::request> requests_;
      Aggregator_t *aggregator_;
  };
  
  template<typename TemplateMap>
  MasterTable<TemplateMap>::MasterTable(boost::mpi::communicator &world,
                                       index_t world_color,
                                       Aggregator_t *aggregator) {
    world_ = world.split(world_color);
    aggregator_=aggregator;
  }

  template<typename TemplateMap>
  void MasterTable<TemplateMap>::InitMasterParams() {
    aggregator_->Init(this->n_entries());
    requests_.resize(this->n_entries());
  }

  template<typename TemplateMap>
  void MasterTable<TemplateMap>::PushToClients(index_t point_id) {
    // check to see if the old request has been received. 
    // If not just cancel it
    if (!requests_[point_id].test()) {
      requests_[point_id].cancel();
    }
    Point_t point;
    this->get(point_id, &point);
    requests_[point_id] = world_.isend(1, 
        point_id, point);
    aggregator_->Reset(point_id, &point);
  }
  
  template<typename TemplateMap>
  void MasterTable<TemplateMap>::PullFromClients(index_t point_id) {
    Point_t point1;
    this->get(point_id, &point1);
    Point_t point;
    point.Copy(point1);
    for(index_t i=1; i< world_.size(); ++i) {
      boost::optional<boost::mpi::status> msg=world_.iprobe(i, point_id);
      if (!msg==false) {
        world_.recv(i, point_id, point);  
      }
      aggregator_->Aggregate(point_id, point, &point1);
    }
  }


}}


#endif

