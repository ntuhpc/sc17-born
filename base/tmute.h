#ifndef tmute_H
#define tmute_H 1
#include "my_operator.h"
#include "oc_float.h"
#include "source_func_3d.h"
/**
   Class to mute a dataset
 */
class tmute : public my_operator {
public:
/**Intialized to tmute function
   \param io - IO for operations
   \param t t0 for mute
   \mod model
   \dat data
 */
tmute(float t, float v, std::shared_ptr<oc_float> mod, std::shared_ptr<oc_float> dat){
	set_domain(mod); set_range(dat);
	t_0=t; vmute=v;
};

/**forward function
   \param bool add to current model
   \param model model
   \param data data
 */
virtual bool forward(bool add,std::shared_ptr<my_vector> model, std::shared_ptr<my_vector>data);
/**adjoint function
   \param bool add to current model
   \param model model
   \param data data
 */
virtual bool adjoint(bool add, std::shared_ptr<my_vector>model,std::shared_ptr<my_vector>data);

/** t0 for mute*/
float t_0;
/** velocity for mute */
float vmute;

};
#endif
