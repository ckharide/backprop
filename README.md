# backprop

# This is to explain How backprop works with a simple function .

The idea in back prop is to compute gradients for a function when the gradient flag is on ( which is generally requires_grad) . 
Lets take a simple example of a func y = x * x and we want to compute the gradients during backprop. 

For computng the gradient the differential rule is rule since so it will be dy/dy * dy/dy => which in this will be 2*x . How does a tensor lib compute the gradients i presumed that it might be using some sort of a mathematical differential lib to compute these gradients but thats not correct. 

# Compute Part 

There are couple of steps involved here 

1. Forward Pass
2. Backward initialization
3. Gradient compute
4. Gradient accumpulate

Lets examine what happens in forward path : Though forward pass is not always computed during backward prop

y._ctx = MultiplyContext(parents=[x, x])
A Multiply context is created based on the operands and it is stored in the _ctx object of y . 




