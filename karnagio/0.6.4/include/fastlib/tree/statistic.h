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
// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file statistic.h
 *
 * Home for the concept of tree statistics.
 *
 * You should define your own statistic that looks like EmptyStatistic.
 *
 * @experimental
 */

#ifndef FASTLIB_TREE_STATISTIC_H
#define FASTLIB_TREE_STATISTIC_H

namespace fl {
namespace tree {

/**
 * Empty statistic if you are not interested in storing statistics
 * in your tree.  Use this as a template for your own.
 *
 * @experimental
 */

class EmptyStatistic {
  public:

    ~EmptyStatistic() {
    }

    /**
     * Initializes by taking statistics on raw data.
     */
    template<typename TreeIterator>
    void Init(TreeIterator &it) {
    }

    /**
     * Initializes by combining statistics of two partitions.
     *
     * This lets you build fast bottom-up statistics when building trees.
     */
    template<typename TreeIterator>
    void Init(TreeIterator &it,
              EmptyStatistic& left_stat, EmptyStatistic& right_stat) {
    }
    /**
     * Serialized the class, writes nothing
     */
    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {

    }
};


class EmptyStatisticMultiBranch {
  public:
    EmptyStatisticMultiBranch() {
    }

    ~EmptyStatisticMultiBranch() {
    }

    /**
     * Initializes by taking statistics on raw data.
     */
    template<typename TreeIterator>
    void Init(TreeIterator& it) {
    }

    /**
     * Initializes by combining statistics of two partitions.
     *
     * This lets you build fast bottom-up statistics when building trees.
     */
    template<typename TreeIterator, typename ContainerType>
    void Init(TreeIterator& it,  const ContainerType &children) {

    }
    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {

    }

};
}; // tree namespace
}; // fl namespace

#endif
