#include "lin_solver.h"
#include "i_op.h"
#include "super_vector.h"
#include "combo_oper.h"


simple_solver::simple_solver(step *s, std::shared_ptr<my_vector>data, std::shared_ptr<my_operator>op, std::shared_ptr<my_operator>wt,
	std::shared_ptr<my_vector>m0){

	init_solve(s);
	rr=init_rr(data,op,wt,m0);
	create_wt_op(op,wt);
	oper=wt_op;
	create_solver_vecs();

}



prec_solver::prec_solver(step *s, std::shared_ptr<my_vector>data, std::shared_ptr<my_operator>op, std::shared_ptr<my_operator>prec, float eps,
	std::shared_ptr<my_operator>wt,std::shared_ptr<my_vector>m0){

	init_solve(s);
	have_op=true;
	std::shared_ptr<my_vector>rtemp=init_rr(data,op,wt,m0);
	std::shared_ptr<my_vector>prec_domain=prec->domain->clone_zero();
	iop.reset(new i_op(prec_domain,prec_domain));
	iop->set_scale(eps);
	create_wt_op(op,wt);
	prec_chain_op.reset(new row_op(prec,wt_op));
	oper.reset(new col_op(prec_chain_op,iop));
	rr=oper->range_vec(rtemp,prec_domain);
	create_solver_vecs();
	pop=prec;

}
reg_solver::reg_solver(step *s, std::shared_ptr<my_vector>data, std::shared_ptr<my_operator>op, std::shared_ptr<my_operator>reg, float eps,
	std::shared_ptr<my_operator>wt, std::shared_ptr<my_vector>m0){

	init_solve(s);
	have_op=true;
	std::shared_ptr<my_vector>rtemp=init_rr(data,op,wt,m0);
	std::shared_ptr<my_vector>reg_domain=reg->range->clone_zero();
	reg->set_scale(eps);
	create_wt_op(op,wt);
	oper.reset(new col_op(wt_op,reg));
	rr=oper->range_vec(rtemp,reg_domain);
	create_solver_vecs();

}
void lin_solver::init_solve(step *s){
	st=s;
	verb=0;
	have_wt=false;
	have_op=false;


}
void lin_solver::create_solver_vecs(){
	fprintf(stderr,"Create solver vecs\n");
	gg=rr->clone_zero();
	m=oper->domain->clone_zero();
	g=oper->domain->clone_zero();
	st->alloc_step(m,rr);

}
void lin_solver::create_wt_op(std::shared_ptr<my_operator>op, std::shared_ptr<my_operator>wt){
	if(wt!=0) {
		wt_op.reset(new row_op(op,wt)); have_wt=true;
	}
	else{
		fprintf(stderr,"in this conditional \n");
		wt_op=op;
	}
}
std::shared_ptr<my_vector>lin_solver::init_rr(std::shared_ptr<my_vector>data, std::shared_ptr<my_operator>op, std::shared_ptr<my_operator>wt, std::shared_ptr<my_vector>m0){

	std::shared_ptr<my_vector>rtemp;
	fprintf(stderr,"Init_rr\n");
	if(wt!=0) {
		if(!wt->domain->check_match(op->range)) {
			fprintf(stderr,"wt and op don't match");
			assert(1==2);
		}
		rtemp=wt->range->clone_zero();
		wt->forward(false,data,rtemp);
		rtemp->scale(-1.);
		if(m0!=0) {
			std::shared_ptr<my_vector>dtemp=data->clone_zero();
			op->forward(false,m0,dtemp);
			wt->forward(true,dtemp,rr);
		}
	}
	else {
		rtemp=data->clone_vec();
		rtemp->scale(-1.);
		if(m0!=0) op->forward(true,m0,rtemp);
	}
	return rtemp;
}
void prec_solver::clean_up(){

}
void lin_solver::clean_up(){

}
void simple_solver::clean_up(){
}
void reg_solver::clean_up(){
}
void lin_solver::set_verbose(int iverb){
	verb=iverb;
}
bool lin_solver::solve(int niter){
	double scale;
	if(verb>0) {
		scale=rr->dot(rr);
		fprintf(stderr,"Initial residual %g scaled to 10000\n",scale);
		scale=100000/scale;
	}

	for(int iter =0; iter < niter; iter++) {
		fprintf(stderr,"Iteration %d / %d\n",iter,niter);
		time_t startitime, enditime;
		startitime = time(&startitime);
		oper->adjoint(false,g,rr,iter);
		oper->forward(false,g,gg,iter);
		double val;

		bool valid=st->step_it(iter,m,g,rr,gg,&val);
		m->info("mod_mov");
		rr->info("rr_mov");
		if(verb>0) {
			fprintf(stderr,"FINISHED ITER %d %g  \n",iter,(double)(val*scale));
		}
		if(!valid) return false;
		enditime = time(&enditime);
		fprintf(stderr,"Iteration %d, wall clock time = %f \n", iter,difftime(enditime, startitime));

	}
	return true;
}
std::shared_ptr<my_vector>lin_solver::return_model(){
	return m->clone_vec();
}
void lin_solver::update_model(std::shared_ptr<my_vector>mod){
	mod->add(m);
}
std::shared_ptr<my_vector>prec_solver::return_model(){
	std::shared_ptr<my_vector>mod=pop->range->clone_zero();
	pop->forward(false,m,mod);


	return mod;
}
void prec_solver::update_model(std::shared_ptr<my_vector>mod){
	pop->forward(true,m,mod);
}
