

data.json: data.P
	Data json=data.P	


wavelet.json: wavelet.P
	Wavelet json=wavelet.P

model: vel.big.json wavelet.json model.P
	Model3D   json=model.P

vel.json: vel.P
	Vel json=vel.P

image: vel.big.json wavelet.json data.big.json model.P
	RTM3D   json=image.P
	

mig.json: vel.json wavelet.json data.json mig.P
	RTM3D json=mig.P 
	
test: vel.json wavelet.json data.json mig.P
	RTM3D json=mig.P 

image.small: vel.small.json wavelet.json data.small.json image.small.P
	RTM3D   json=image.small.P
	
image.small: vel.small.json wavelet.json data.small.json model.P
	RTM3D   json=image.small.P
	
	
model.small: vel.small.json wavelet.json data.small.json model.small.P
	Model3D   json=model.small.P
	
debug: vel.json wavelet.json data.json mig.P
	valgrind RTM3D json=mig.P 
	

