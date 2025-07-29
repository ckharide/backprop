# backprop

# This is to explain How backprop works with a simple function .

The idea in back prop is to compute gradients for a function when the gradient flag is on ( which is generally requires_grad) . 
Lets take a simple example of a func y = x * x and we want to compute the gradients during backprop. 

For computng the gradient the differential rule is rule since so it will be dy/dy * dy/dy => which in this will be 2*x . How does a tensor lib compute the gradients i presumed that it might be using some sort of a mathematical differential lib to compute these gradients but thats not correct. 




