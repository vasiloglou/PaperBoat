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

#ifndef FL_LITE_INCLUDE_FASTLIB_COMMUNICATION_CONNECTION_H_
#define FL_LITE_INCLUDE_FASTLIB_COMMUNICATION_CONNECTION_H_

#include "boost/asio.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/archive/text_oarchive.hpp"
#include "boost/bind.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/tuple/tuple.hpp"
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>

namespace fl { namespace com {
  /// The connection class provides serialization primitives on top of a socket.
  /**
   * Each message sent using this class consists of:
   * @li An 8-byte header containing the length of the serialized data in
   * hexadecimal.
   * @li The serialized data.
   */
  class connection {
    public:
      /// Constructor.
      connection(boost::asio::io_service& io_service)
          : socket_(io_service) {
       
          // boost::asio::socket_base::debug option1(true);
          // try {
          //  socket_.set_option(option1);
          // }
          // catch(const std::exception &ex) {
          //  fl::logger->Message()<<ex.what();
          // }
      }
      
      void Close() {
        socket_.close();
      }
      /// Get the underlying socket. Used for making a connection or for accepting
      /// an incoming connection.
      boost::asio::ip::tcp::socket& socket() {
        return socket_;
      }
      template<typename T, typename Handler>
      void sync_write(T &t, Handler handler) {
        // Serialize the data first so we know how large it is.
        std::ostringstream archive_stream;
        boost::archive::text_oarchive archive(archive_stream);
        archive << t;
        outbound_data_ = archive_stream.str();
        // Format the header.
        std::ostringstream header_stream;
        header_stream << std::setw(header_length)
          << std::hex << outbound_data_.size();
        if (!header_stream || header_stream.str().size() != header_length) {
          // Something went wrong, inform the caller.
          boost::system::error_code error(boost::asio::error::invalid_argument);
          socket_.io_service().post(boost::bind(handler, error, 1));
          return;
        }
        outbound_header_ = header_stream.str();
    
        // Write the serialized data to the socket. We use "gather-write" to send
        // both the header and the data in a single write operation.
        std::vector<boost::asio::const_buffer> buffers;
        buffers.push_back(boost::asio::buffer(outbound_header_));
        buffers.push_back(boost::asio::buffer(outbound_data_));
        boost::asio::write(socket_, buffers);       
      } 
      template<typename T>
      void sync_write(T &t) {
         sync_write(t, 
             boost::bind(&connection::handle, 
               this, 
               boost::asio::placeholders::error, this));
      } 
      template<typename T, typename Handler>
      void sync_read(T &t, Handler handler) {
        boost::asio::read(socket_, boost::asio::buffer(inbound_header_));
        // Determine the length of the serialized data.
        std::istringstream is(std::string(inbound_header_, header_length));
        std::size_t inbound_data_size = 0;
        if (!(is >> std::hex >> inbound_data_size)) {
          // Header doesn't seem to be valid. Inform the caller.
          boost::system::error_code error(boost::asio::error::invalid_argument);
          handler(error, this);
          return;
        }

        // Start a synchronous call to receive the data.
        inbound_data_.resize(inbound_data_size);
        boost::asio::read(socket_, boost::asio::buffer(inbound_data_));
        try {
           std::string archive_data(&inbound_data_[0], inbound_data_.size());
           std::istringstream archive_stream(archive_data);
           boost::archive::text_iarchive archive(archive_stream);
           archive >> t;
         }
         catch (std::exception& e) {
           // Unable to decode data.
           boost::system::error_code error(boost::asio::error::invalid_argument);
          handler(error, this);
           return;
         }
      } 

      template<typename T>
      void sync_read(T &t) {
         sync_read(t, 
             boost::bind(&connection::handle, 
               this, 
               boost::asio::placeholders::error, this));
      } 

      /// Asynchronously write a data structure to the socket.
      template <typename T, typename Handler>
      void async_write(const T& t, Handler handler) {
        // Serialize the data first so we know how large it is.
        std::ostringstream archive_stream;
        boost::archive::text_oarchive archive(archive_stream);
        archive << t;
        outbound_data_ = archive_stream.str();
    
        // Format the header.
        std::ostringstream header_stream;
        header_stream << std::setw(header_length)
          << std::hex << outbound_data_.size();
        if (!header_stream || header_stream.str().size() != header_length) {
          // Something went wrong, inform the caller.
          boost::system::error_code error(boost::asio::error::invalid_argument);
          socket_.io_service().post(boost::bind(handler, error));
          return;
        }
        outbound_header_ = header_stream.str();
    
        // Write the serialized data to the socket. We use "gather-write" to send
        // both the header and the data in a single write operation.
        std::vector<boost::asio::const_buffer> buffers;
        buffers.push_back(boost::asio::buffer(outbound_header_));
        buffers.push_back(boost::asio::buffer(outbound_data_));
        boost::asio::async_write(socket_, buffers, handler);
      }
 
      template<typename T>
      void async_write(T &t) {
         async_write(t, 
             boost::bind(&connection::handle, 
               this, 
               boost::asio::placeholders::error, this));
      } 

      /// Asynchronously read a data structure from the socket.
      template <typename T, typename Handler>
      void async_read(T& t, Handler handler) {
        // Issue a read operation to read exactly the number of bytes in a header.
        void (connection::*f)(
            const boost::system::error_code&,
            T&, boost::tuple<Handler>)
          = &connection::handle_read_header<T, Handler>;
        boost::asio::async_read(socket_, boost::asio::buffer(inbound_header_),
            boost::bind(f,
              this, boost::asio::placeholders::error, boost::ref(t),
              boost::make_tuple(handler)));
      }
 
      template<typename T>
      void async_read(T &t) {
         async_read(t, 
             boost::bind(&connection::handle, 
               this, 
               boost::asio::placeholders::error, this));
      } 

   
      /// Handle a completed read of a message header. The handler is passed using
      /// a tuple since boost::bind seems to have trouble binding a function object
      /// created using boost::bind as a parameter.
      template <typename T, typename Handler>
      void handle_read_header(const boost::system::error_code& e,
          T& t, boost::tuple<Handler> handler) {
        if (e) {
          boost::get<0>(handler)(e);
        } else {
          // Determine the length of the serialized data.
          std::istringstream is(std::string(inbound_header_, header_length));
          std::size_t inbound_data_size = 0;
          if (!(is >> std::hex >> inbound_data_size)) {
            // Header doesn't seem to be valid. Inform the caller.
            boost::system::error_code error(boost::asio::error::invalid_argument);
            boost::get<0>(handler)(error);
            return;
          }
    
          // Start an asynchronous call to receive the data.
          inbound_data_.resize(inbound_data_size);
          void (connection::*f)(
              const boost::system::error_code&,
              T&, boost::tuple<Handler>)
            = &connection::handle_read_data<T, Handler>;
          boost::asio::async_read(socket_, boost::asio::buffer(inbound_data_),
            boost::bind(f, this,
              boost::asio::placeholders::error, boost::ref(t), handler));
        }
      }
    
      /// Handle a completed read of message data.
      template <typename T, typename Handler>
      void handle_read_data(const boost::system::error_code& e,
          T& t, boost::tuple<Handler> handler) {
        if (e) {
          boost::get<0>(handler)(e);
        } else {
          // Extract the data structure from the data just received.
          try {
            std::string archive_data(&inbound_data_[0], inbound_data_.size());
            std::istringstream archive_stream(archive_data);
            boost::archive::text_iarchive archive(archive_stream);
            archive >> t;
          }
          catch (std::exception& e) {
            // Unable to decode data.
            boost::system::error_code error(boost::asio::error::invalid_argument);
            boost::get<0>(handler)(error);
            return;
          }
    
          // Inform caller that data has been received ok.
          boost::get<0>(handler)(e);
        }
      }
      void handle(const boost::system::error_code &e, connection *conn) {
      
      }
    
    private:
      /// The underlying socket.
      boost::asio::ip::tcp::socket socket_;
    
      /// The size of a fixed length header.
      enum { header_length = 8 };
    
      /// Holds an outbound header.
      std::string outbound_header_;
    
      /// Holds the outbound data.
      std::string outbound_data_;
    
      /// Holds an inbound header.
      char inbound_header_[header_length];
    
      /// Holds the inbound data.
      std::vector<char> inbound_data_;
    };
    
    typedef boost::shared_ptr<connection> connection_ptr;

}} // namespaces

#endif

