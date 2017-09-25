#include "combo_oper.h"
#include <cassert>

col_op::col_op(std::shared_ptr<my_operator>op){

	ops.push_back(op);
	create_super();

}
col_op::col_op(std::shared_ptr<my_operator>op1,std::shared_ptr<my_operator>op2){

	ops.push_back(op1);
	ops.push_back(op2);
	create_super();
}
col_op::col_op(std::vector<std::shared_ptr<my_operator> >op){

	for(int i=0; i < (int)op.size(); i++) ops.push_back(op[i]);
	create_super();

}
void col_op::add_op(std::shared_ptr<my_operator>op){

	ops.push_back(op);
	create_super();
}
void col_op::add_ops(std::vector<std::shared_ptr<my_operator> >os){

	for(int i=0; i < (int) os.size(); i++) ops.push_back(os[i]);
	create_super();

}
void col_op::create_super(){

	std::vector < std::shared_ptr<my_vector> > ds,rs;
	for(int i=0; i < (int)ops.size(); i++) rs.push_back(ops[i]->range);
	for(int i=1; i < (int)ops.size(); i++) {
		if(!ops[0]->domain->check_match(ops[i]->domain)) {
			fprintf(stderr,"ERROR:Domain does not match in array op 0 and %d \n",i);
		}
	}

	fprintf(stderr,"range vector combo \n");
	range.reset( new super_vector(rs,true));
	domain=ops[0]->domain->clone_space();


}
bool col_op::forward(bool add, std::shared_ptr<my_vector>model, std::shared_ptr<my_vector>data){

	if(!check_domain_range(model,data)) {
		fprintf(stderr,"col op match failed(1)\n");
		assert(1==0);

		return false;
	}
	std::shared_ptr<super_vector> d= std::dynamic_pointer_cast<super_vector>( data);

	for(int i=0; i < ops.size(); i++) {

		if(!ops[i]->forward(add,model,d->vecs[i])) {
			fprintf(stderr,"failed running array op %d \n",i);
			assert(1==0);
			return false;
		}

	}
	return true;
}
bool col_op::adjoint(bool add, std::shared_ptr<my_vector>model, std::shared_ptr<my_vector>data){

	if(!check_domain_range(model,data)) {
		fprintf(stderr,"col op match failed(2)\n");
		assert(1==0);

		return false;
	}

	std::shared_ptr<super_vector> d= std::dynamic_pointer_cast<super_vector>( data);
	bool a=add;
	for(int i=0; i < ops.size(); i++) {

		if(!ops[i]->adjoint(a,model,d->vecs[i])) {
			fprintf(stderr,"failed running array op %d \n",i);
			assert(1==0);

			return false;
		}

		a=true;
	}
	return true;
}
diag_op::diag_op(std::shared_ptr<my_operator>op){
	ops.push_back(op);
	create_super();

}
diag_op::diag_op(std::shared_ptr<my_operator>op1,std::shared_ptr<my_operator>op2){
	ops.push_back(op1);
	ops.push_back(op2);
	create_super();
}
diag_op::diag_op(std::vector<std::shared_ptr<my_operator> >op){
	for(int i=0; i < (int)op.size(); i++) ops.push_back(op[i]);
	create_super();

}
void diag_op::add_op(std::shared_ptr<my_operator>op){

	ops.push_back(op);
	create_super();
}
void diag_op::add_ops(std::vector<std::shared_ptr<my_operator> >os){

	for(int i=0; i < (int) os.size(); i++) ops.push_back(os[i]);
	create_super();

}
void diag_op::create_super(){
	std::vector < std::shared_ptr<my_vector> > ds,rs;
	for(int i=0; i < (int)ops.size(); i++) {
		ds.push_back(ops[i]->domain);
		rs.push_back(ops[i]->range);
	}

	domain.reset(new super_vector(ds,true));
	range.reset( new super_vector(rs,true));

}
bool diag_op::forward(bool add, std::shared_ptr<my_vector>model, std::shared_ptr<my_vector>data){
	if(!check_domain_range(model,data)) {
		fprintf(stderr,"diag op match failed(3)\n");
		return false;
	}
	std::shared_ptr<super_vector> m= std::dynamic_pointer_cast<super_vector>(model);
	std::shared_ptr<super_vector> d= std::dynamic_pointer_cast<super_vector>( data);
	for(int i=0; i < ops.size(); i++) {


		if(!ops[i]->forward(add,m->vecs[i],d->vecs[i])) {
			fprintf(stderr,"failed running array op %d \n",i);
			return false;
		}
	}
	return true;
}
bool diag_op::adjoint(bool add, std::shared_ptr<my_vector>model, std::shared_ptr<my_vector>data){

	if(!check_domain_range(model,data)) {
		fprintf(stderr,"diag op match failed(3)\n");
		return false;
	}
	std::shared_ptr<super_vector> m= std::dynamic_pointer_cast<super_vector>(model);
	std::shared_ptr<super_vector> d= std::dynamic_pointer_cast<super_vector>( data);
	for(int i=0; i < ops.size(); i++) {

		if(!ops[i]->adjoint(add,m->vecs[i],d->vecs[i])) {
			fprintf(stderr,"failed running array op %d \n",i);
			return false;
		}
	}
	return true;
}
bool row_op::forward(bool add, std::shared_ptr<my_vector>model, std::shared_ptr<my_vector>data){
	if(!check_domain_range(model,data)) {
		fprintf(stderr,"row op match failed\n");
		return false;
	}
	std::shared_ptr<my_vector>d,r,dom=model,ran;
	bool a;

	for(int i=0; i < (int) ops.size(); i++) {
		if(i!=(int)ops.size()-1) {
			a=false;
			r=ops[i]->range->clone_zero();
			ran=r;
		}
		else {
			a=add;
			ran=data;
		}
		if(!ops[i]->forward(a,dom,ran)) {
			fprintf(stderr,"failed running row op %d \n",i);
			return false;
		}

		dom=ran;

	}
	return true;

}
bool row_op::adjoint(bool add, std::shared_ptr<my_vector>model, std::shared_ptr<my_vector>data){

	if(!check_domain_range(model,data)) {
		fprintf(stderr,"array op match failed\n");
		return false;
	}
	std::shared_ptr<my_vector>dom,d,ran=data;;
	bool a=false;
	for(int i=(int)ops.size()-1; i >=0; i--) {

		if(i!=0) {
			a=false;
			d=ops[i]->domain->clone_zero();
			dom=d;
		}
		else {
			a=add;
			dom=model;
		}
		bool x=ops[i]->adjoint(a,dom,ran);
		if(!x) {
			fprintf(stderr,"failed running chain op %d \n",i);
			return false;
		}
		ran=dom;

	}

	return true;
}

std::shared_ptr<my_vector> col_op::range_vec(std::shared_ptr<my_vector>v1, std::shared_ptr<my_vector>v2){

	std::shared_ptr<super_vector > vec(new super_vector(v1,v2));
	if(!vec->check_match(range)) {
		fprintf(stderr,"range vector doesn't match suplied vecs");
		assert(1==2);
	}
	return vec;
}
std::shared_ptr<my_vector> diag_op::domain_vec(std::shared_ptr<my_vector>v1,std::shared_ptr<my_vector>v2){

	std::shared_ptr<super_vector >vec(new super_vector(v1,v2));
	if(!vec->check_match(domain)) {
		fprintf(stderr,"range vector doesn't match suplied vecs");
		assert(1==2);
	}
	return vec;
}
std::shared_ptr<my_vector> diag_op::range_vec(std::shared_ptr<my_vector>v1, std::shared_ptr<my_vector>v2){

	std::shared_ptr<super_vector >vec(new super_vector(v1,v2));
	if(!vec->check_match(range)) {
		fprintf(stderr,"range vector doesn't match suplied vecs");
		assert(1==2);
	}
	return vec;
}

void row_op::chain_init(std::vector<std::shared_ptr<my_operator> > os){

	if(!check_valid_chain()) {
		fprintf(stderr,"trouble creating chain op \n");
		return;
	}
	for(int i=0; i< (int)os.size(); i++) ops.push_back(os[i]);
	domain=os[0]->domain->clone_space();
	range=os[os.size()-1]->range->clone_space();
}
bool row_op::check_valid_chain(){

	for(int i=0; i < (int) ops.size()-1; i++) {
		if(!ops[i]->range->check_match(ops[i+1]->domain)) {
			fprintf(stderr,"domain and range don't match when creating super operator (%d %d) \n",i,i+1);
			return false;
		}
	}
	return true;
}
void row_op::add_op(std::shared_ptr<my_operator>op){
	ops.push_back(op);
	if(!check_valid_chain())
		fprintf(stderr,"trouble adding operator\n");

}
