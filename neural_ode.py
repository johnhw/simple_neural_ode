# This implements the ideas in the paper "Neural Ordinary Differential Equations"
# in as simple a form as possible, using only autograd. It is not efficient.
# It is not useful for any practical purpose. 
# Use [torchdiffeq](https://github.com/rtqichen/torchdiffeq) for any real use.
#
# > [1] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. 
# "Neural Ordinary Differential Equations." *Advances in Neural Processing Information Systems.* 2018. 
# [[arxiv]](https://arxiv.org/abs/1806.07366)
#
# The implementation is based on the 
# [write up of Per Vognsen](https://gist.github.com/pervognsen/3bac77cff45cfa7378c1a6d3bedf61d6)
# in terms of the costate vector, which is a clear exposition of how the adjoint method works.

import scipy.integrate
import autograd.numpy as np
from autograd import jacobian, grad
from autograd.misc.flatten import flatten

odeint = scipy.integrate.odeint


################################################################################

# Define the forward and backward (adjoint) dynamics


def forward(x0, f, t0, t1):
    # return only the result at t1, since we know t0
    return odeint(f, y0=x0, t=[t0, t1])[1]


def backward(x1, dloss_dx_t1, f, df_dx, t0, t1):
    x_partition = len(x1)
    split = lambda xp: (xp[:x_partition], xp[x_partition:])
    join = lambda x, p: np.concatenate([x, p])

    def augmented_dynamics(xp, t):
        # split the augmented vector [x | p] back into x and p
        x_in, p_in = split(xp)
        # [f(x,t) | -p(t) f'(x,t)]
        return join(f(x_in, t), -p_in @ df_dx(x_in, t))

    # get derivatives at time t=0, using the integrator
    dx0, dp0 = split(forward(join(x1, dloss_dx_t1), augmented_dynamics, t1, t0))
    return dp0


################################################################################

# Compute the gradient of the loss w.r.t weights, given a loss function
# at the output and the ODE function


def make_grad_loss(loss_fn, layer_fn, weights):

    f = lambda x, t, w=weights: layer_fn(x, t, w)
    # with respect to x
    df_dx = lambda x, t, w=weights: jacobian(layer_fn)(x, t, w)
    # with respect to w
    df_dw = lambda x, t, w=weights: jacobian(layer_fn, 2)(x, t, w)
    # compute gradient of loss function w.r.t output/state
    dloss_dx = grad(loss_fn)

    def grad_loss(x0, x1, t0, t1):
        x1_pred = forward(x0, f, t0, t1)  # predict to t1
        loss = loss_fn(x1_pred, x1)  # loss at prediction
        # compute gradient of loss of (x1_pred, x1)
        dloss_dx_t1 = dloss_dx(x1_pred, x1)
        # propagate covector backwards to compute dloss_dx at time t0
        dloss_dx_t0 = backward(x1_pred, dloss_dx_t1, f, df_dx, t0, t1)
        dloss_dw = df_dw(x0, 0).T @ dloss_dx_t0  # chain rule
        return loss, dloss_dw

    return grad_loss


################################################################################
# define a basic tanh -> linear layer


def random_init(n):
    # create random weight matrix
    # input is [x,t], output is same dimension as x
    w1 = np.random.normal(0, 0.1, (n + 1, n + 1))
    w2 = np.random.normal(0, 0.1, (n, n + 1))
    b1 = np.random.normal(0, 0.1, (n + 1))
    b2 = np.random.normal(0, 0.1, (n))
    return flatten([w1, w2, b1, b2])


def tanh_layer(x, w, unflatten):
    # compute activation for linear(tanh())
    w1, w2, b1, b2 = unflatten(w)
    return w2 @ np.tanh(w1 @ x + b1) + b2


def make_layer(n, init_fn=random_init, layer_fn=tanh_layer):
    # weights assume a vector [x,t], where x has n elements, and t is a scalar
    weights, unflatten = init_fn(n)

    def layer(x, t, w):
        # reshape to matrix, combine with t and bias
        # then truncate to original dimension and apply activation
        xtb = np.concatenate([x, [t]])
        return layer_fn(xtb, w, unflatten)

    return layer, weights


################################################################################

# Straightforward gradient descent


def gradient_descent(x0s, x1s, t0s, t1s, weights, grad_loss, steps, delta=1e-2):
    f = lambda x, t, w=weights: layer(x, t, w)

    # standard gradient descent
    for i in range(steps + 1):
        total_loss, total_grad = 0, np.zeros_like(weights)

        # (very slowly) accumulate gradient
        for x0, x1, t0, t1 in zip(x0s, x1s, t0s, t1s):
            l, dl_dw = grad_loss(x0, x1, t0, t1)
            total_grad += dl_dw
            total_loss += l

        weights -= delta * total_grad

        if i % 20 == 0:
            print("{i:8d} Loss {total_loss:.4f}".format(i=i, total_loss=total_loss))

    return weights


################################################################################
# Simple test: fit a rotational vector field

if __name__ == "__main__":
    # 10 random points in 3D
    x0s = np.random.normal(0, 0.5, (10, 3))

    # Learn a 90 degree rotation about the origin
    x1s = x0s @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    t0, t1 = 0, 1
    layer, weights = make_layer(3)
    grad_loss = make_grad_loss(
        loss_fn=lambda x, y: np.sum((x - y) ** 2), layer_fn=layer, weights=weights
    )

    gradient_descent(
        x0s=x0s,
        x1s=x1s,
        t0s=np.full(10, 0),
        t1s=np.full(10, 1),
        weights=weights,
        grad_loss=grad_loss,
        steps=500,
        delta=2e-2,
    )

