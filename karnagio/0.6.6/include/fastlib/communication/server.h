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

#ifndef FL_LITE_INCLUDE_FASTLIB_COMMUNICATION_SERVER_H_
#define FL_LITE_INCLUDE_FASTLIB_COMMUNICATION_SERVER_H_

#include "boost/asio.hpp"
#include "boost/bind.hpp"
#include "boost/lexical_cast.hpp"
#include <iostream>
#include "fastlib/table/default_table.h"
#include "fastlib/table/table_dev.h"
#include "fastlib/data/multi_dataset_dev.h"
#include "connection.h"

namespace fl { namespace com {

  /// Serves stock quote information to any client that connects to it.
  template<typename ResponderType>
  class Server {
    public:
      /// Constructor opens the acceptor and starts waiting for the first incoming
      /// connection.
      Server(boost::asio::io_service& io_service, 
          const std::string &port,
          ResponderType *responder)
        : acceptor_(io_service,
            boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 
             boost::lexical_cast<unsigned short>(port))),
            responder_(responder)
      {
         // boost::asio::socket_base::enable_connection_aborted option2(true);
         // try {
         //   acceptor_.set_option(option2);
         // } 
         // catch(const std::exception &ex) {
         //   fl::logger->Message()<<ex.what();
         //  }
        // Start an accept operation for a new connection.
        connection_ptr new_conn(new connection(acceptor_.io_service()));
        acceptor_.async_accept(new_conn->socket(),
            boost::bind(&Server::handle_accept, this,
              boost::asio::placeholders::error, new_conn));
      }
      ~Server() {} 
      void Close() {
        acceptor_.close();
      }
      /// Handle completion of a accept operation.
      void handle_accept(const boost::system::error_code& e, connection_ptr conn) {
        if (!e) {
           if (responder_->operator()(e, conn)==true) {  
             // Start an accept operation for a new connection.
             connection_ptr new_conn(new connection(acceptor_.io_service()));
             acceptor_.async_accept(new_conn->socket(),
                boost::bind(&Server::handle_accept, this,
                  boost::asio::placeholders::error, new_conn));
           } else {
             acceptor_.close();
           }
        } else {
          // An error occurred. Log it and return. Since we are not starting a new
          // accept operation the io_service will run out of work to do and the
          // server will exit.
          fl::logger->Die() << "Server error accepting connection impossible" <<
            ", error: " <<e.message();
        }
      }
    
      /// Handle completion of a write operation.
      void handle_write(const boost::system::error_code& e, connection_ptr conn) {
        // Nothing to do. The socket will be closed automatically when the last
        // reference to the connection object goes away.
      }
    
    private:
      /// The acceptor object used to accept incoming socket connections.
      boost::asio::ip::tcp::acceptor acceptor_;
    
      ResponderType *responder_;
  };
  
}}


#endif 
