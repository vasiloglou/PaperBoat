#ifndef FL_LITE_MLPACK_ALLKN_ALLKN_INST_H
#define FL_LITE_MLPACK_ALLKN_ALLKN_INST_H
namespace fl {
namespace ml {
class Instantiator {
  public:
    struct Operator {
      template<typename T>
      void operator()(T) ;
    };
    Instantiator();
    void DummyFunction();
};
}
}

#endif
