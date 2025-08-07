# backprop

# This is to explain How backprop works with a simple function .

The idea in back prop is to compute gradients for a function when the gradient flag is on ( which is generally requires_grad) . 
Lets take a simple example of a func y = x * a and we want to compute the gradients during backprop. 

For computng the gradient the differential rule is rule w.r.t x since so it will be dy/dy * dy/dx => which in this will be 1*a => a . How does a tensor lib compute the gradients i presumed that it might be using some sort of a mathematical differential lib to compute these gradients but thats not correct. 

# Compute Part 

There are couple of steps involved here 

1. Forward Pass
2. Backward initialization
3. Gradient compute
4. Gradient accumulate

     # 1. Lets examine what happens in forward path : Though forward pass is not always computed during backward prop

      + y._ctx = MultiplyContext(parents=[x, a])
      + A Multiply context is created based on the operands and it is stored in the _ctx object of y . 
      + A graph is built with the nodes x and a to compute the gradients later. It knows which operands have contributed to the multiply operation .

      # 2. Backward initialization
       Backward initializaton happens when y.backward() is called. 

      # 3. Gradient compute 

      + Call Multiply backward: returns (grad_wrt_a = 1*x, grad_wrt_x = 1*a)
      + This happens via a multiply rule which knows how to compute the gradients via a differential rule
      
      ``` python
      def backward(ctx, grad_output):
        a, x = ctx.parents
      return grad_output * x, grad_output * a
      ```
      Essentially what this code does is if you want to compute the dy/da it returns x and if you want to compute dy/dx it returns a which is correct.

      # Gradient accumulate
        Once all the gradients are computed they are accumulated and added to the existing gradients this is not needed in this example but makes sense when the           variable input is part of multiple functions for ex :
        y = x * a
        z = W*x + b

        When you call loss.backward(), it adds gradient to each parameter's .grad, like:
        self.grad += new_gradient

      









