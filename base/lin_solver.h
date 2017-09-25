#ifndef BASIC_SOLVER_H
#include "my_vector.h"
#include "my_operator.h"
#include "cgstep.h"
#include <vector>


class lin_solver {
public:
lin_solver(){
}



~lin_solver(){
	clean_up();
}
void set_verbose(int iv);
bool solve(int niter);
void init_solve(step *s);



void create_wt_op(std::shared_ptr<my_operator>op, std::shared_ptr<my_operator>wt);
void create_solver_vecs();
std::shared_ptr<my_vector>init_rr(std::shared_ptr<my_vector>data, std::shared_ptr<my_operator>op, std::shared_ptr<my_operator>wt, std::shared_ptr<my_vector>m0);
virtual void update_model(std::shared_ptr<my_vector>mod);
virtual std::shared_ptr<my_vector>return_model();


std::shared_ptr<my_vector > g,m,gg,rr;
std::shared_ptr<my_operator >oper,wt_op,iop;
int verb;
bool have_op,have_wt;
step *st;

private:
void clean_up();
double scale;




};
class simple_solver : public lin_solver {
public:
simple_solver(){
}
simple_solver(step *s,std::shared_ptr<my_vector>data, std::shared_ptr<my_operator>op,
	std::shared_ptr<my_operator>wt=nullptr, std::shared_ptr<my_vector>m0=nullptr);
~simple_solver(){
};


private:
void clean_up();

};
class reg_solver : public lin_solver {
public:
reg_solver(){
};
reg_solver(step *s,std::shared_ptr<my_vector>data, std::shared_ptr<my_operator>op,std::shared_ptr<my_operator>reg, float eps,
	std::shared_ptr<my_operator>wt=nullptr, std::shared_ptr<my_vector>m=nullptr);

~reg_solver(){
};

private:
void clean_up();
};
class prec_solver : public lin_solver {
public:
prec_solver(){
};
prec_solver(step *s,std::shared_ptr<my_vector>data, std::shared_ptr<my_operator>op,std::shared_ptr<my_operator>reg, float eps,
	std::shared_ptr<my_operator>wt=nullptr, std::shared_ptr<my_vector>m0=nullptr);

~prec_solver(){
	clean_up();
}
virtual void update_model(std::shared_ptr<my_vector>mod);
virtual std::shared_ptr<my_vector>return_model();

private:
void clean_up();
std::shared_ptr<my_operator>pop,prec_chain_op;
};
#endif
