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

#ifndef FL_LITE_INCLUDE_FASTLIB_COMMUNICATION_CLIENT_H_
#define FL_LITE_INCLUDE_FASTLIB_COMMUNICATION_CLIENT_H_
#include "boost/asio.hpp"
#include "boost/bind.hpp"
#include <iostream>
#include <vector>
#include "connection.h" // Must come before boost/serialization headers.
#include "boost/serialization/vector.hpp"

namespace fl { namespace com {
  template<typename ResponderType>
  class Client {
    public:
      /// Constructor starts the asynchronous connect operation.
      Client(boost::asio::io_service& io_service,
           const std::string& host, 
           const std::string& service,
           ResponderType *responder)
        : connection_(new connection(io_service)), responder_(responder) {
        // Resolve the host name into an IP address.
        boost::asio::ip::tcp::resolver resolver(io_service);
        boost::asio::ip::tcp::resolver::query query(host, service);
        boost::asio::ip::tcp::resolver::iterator endpoint_iterator =
            resolver.resolve(query);
        boost::asio::ip::tcp::endpoint endpoint = *endpoint_iterator;

        // Start an asynchronous connect operation.
        connection_->socket().async_connect(endpoint,
          boost::bind(&Client::handle_connect, this,
          boost::asio::placeholders::error, ++endpoint_iterator));
      }

      /// Handle completion of a connect operation.
      void handle_connect(const boost::system::error_code& e,
          boost::asio::ip::tcp::resolver::iterator endpoint_iterator) {
        if (!e) {
          // Successfully established connection. 
          (*responder_)(e, connection_);
        } else if (endpoint_iterator != boost::asio::ip::tcp::resolver::iterator()) {
          // Try the next endpoint.
          connection_->socket().close();
          boost::asio::ip::tcp::endpoint endpoint = *endpoint_iterator;
          connection_->socket().async_connect(endpoint,
          boost::bind(&Client::handle_connect, this,
            boost::asio::placeholders::error, ++endpoint_iterator));
        } else {
          // An error occurred. Log it and return. Since we are not starting a new
          // operation the io_service will run out of work to do and the client will
          // exit.
          fl::logger->Die() << e.message();
        }
      }

      // Since we are not starting a new operation the io_service will run out of
      // work to do and the client will exit.

    private:
      // The connection to the server.
      connection_ptr connection_;
      ResponderType *responder_;
  };

}}

#endif
