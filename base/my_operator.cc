#include <my_operator.h>
#include <math.h>
#include <stdio.h>

void my_operator::set_domain(std::shared_ptr<my_vector>dom){
	scale=1;
	domain=dom->clone_space();
}
void my_operator::set_range(std::shared_ptr<my_vector>ran){
	scale=1.;
	range=ran->clone_space();

}
bool my_operator::dot_test(bool verb){
	std::shared_ptr<my_vector>  mod1,mod2,dat1,dat2;

	mod1=domain->clone_vec();
	mod2=mod1->clone_vec();
	dat1=range->clone_vec();
	dat2=dat1->clone_vec();
	mod1->random();
	dat2->random();




	// fprintf(stderr,"INITIAL DATA\n");
//   mod1->info("Input mod",1);
//   dat2->info("Input dat",2);

	forward(false,mod1,dat1);
	double dot1=dat1->dot(dat2);



	adjoint(false,mod2,dat2);
	double dot2=mod2->dot(mod1);


	if(verb) fprintf(stderr,"Dot(add=false) %g %g \n",dot1,dot2);


	forward(true,mod1,dat1);
	adjoint(true,mod2,dat2);


//  mod2->info("OUT mod",1);
	// dat1->info("OUR dat",2);


	double dot3=mod1->dot(mod2);
	double dot4=dat1->dot(dat2);
	if(verb) fprintf(stderr,"Dot(add=true) %g %g \n",dot4,dot3);



	if(fabs(dot1) < 1e-12 || fabs(dot3) < 1e-12) {
		fprintf(stderr,"Dot product suspiciously small\n");

		return false;
	}



	if(fabs( (dot1-dot2)/dot1) >.0001) {
		fprintf(stderr,"Failed add=false dot product %g %g\n",dot1,dot2);
		return false;
	}

	if(fabs( (dot3-dot4)/dot3) >.0001) {
		fprintf(stderr,"Failed add=true dot product  %g (%g %g) \n",fabs( (dot3-dot4)/dot3),dot3,dot4);
		return false;
	}
	return true;

}
void my_operator::hessian(std::shared_ptr<my_vector>in,std::shared_ptr<my_vector>out){

	std::shared_ptr<my_vector> d=range->clone_vec();
	forward(false,in,d);
	adjoint(false,out,d);

}
bool my_operator::check_domain_range(std::shared_ptr<my_vector>dom, std::shared_ptr<my_vector>rang){
	if(!dom->check_match(domain)) {
		fprintf(stderr,"domains don't match\n");
		return false;
	}
	if(!rang->check_match(range)) {
		fprintf(stderr,"ranges don't match\n");
		return false;
	}
	return true;
}
