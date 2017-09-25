#ifndef I_OP_H
#define I_OP_H 1
#include "my_operator.h"
class i_op : public my_operator {
public:
i_op(std::shared_ptr<my_vector>mod, std::shared_ptr<my_vector>dat);
virtual bool forward(bool add, std::shared_ptr<my_vector> model, std::shared_ptr<my_vector>data);
virtual bool adjoint(bool add, std::shared_ptr<my_vector>model, std::shared_ptr<my_vector>data);






};
#endif
