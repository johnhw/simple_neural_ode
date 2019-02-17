# simple_neural_ode
Simple, basic implementation of neural ODE in autograd/numpy. For learning only

This implements the ideas in the paper "Neural Ordinary Differential Equations"
in as simple a form as possible, using only autograd. It is not efficient.
It is not useful for any practical purpose. 
Use [torchdiffeq](https://github.com/rtqichen/torchdiffeq) for any real use.

> [1] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. 
> "Neural Ordinary Differential Equations." *Advances in Neural Processing Information Systems.* 2018. 
> [[arxiv]](https://arxiv.org/abs/1806.07366)

 The implementation is based on the 
 [write up of Per Vognsen](https://gist.github.com/pervognsen/3bac77cff45cfa7378c1a6d3bedf61d6)
 in terms of the costate vector, which is a clear exposition of how the adjoint method works.
