from dataclasses import dataclass
from typing import Any, Iterable, Tuple, List

from collections import defaultdict  # importing deque

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # f'(x0) ≈ (f(x0 + h) - f(x0 - h)) / (2h)

    def derivative_small(new: float) -> Any:
        subList = list(vals)
        subList[arg] = new

        return f(*subList)

    return (derivative_small(vals[arg] + epsilon) - derivative_small(vals[arg] - epsilon)) / (2 * epsilon)
    # raise NotImplementedError("Need to implement for Task 1.1")


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")

    def visit(node: Variable) -> None:
        if node.unique_id in seen or node.is_constant():
            return
        for p in node.parents:
            visit(p)

        seen.add(node.unique_id)
        res.append(node)
    res: List[Variable] = []
    seen: set[float] = set()
    visit(variable)
    return res


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.
    """
    # TODO: Implement for Task 1.4.
    ordered_queue = list(topological_sort(variable))
    scalar_derivatives = defaultdict(float)
    scalar_derivatives[variable.unique_id] = deriv

    while ordered_queue:
        curr = ordered_queue.pop()
        if not curr.is_leaf():
            derivat = curr.chain_rule(d_output=scalar_derivatives[curr.unique_id])  # call chain rule if not leaf
            for scalar, der in derivat:  # loop through all scalars from chain rule output
                scalar_derivatives[scalar.unique_id] += der
        if curr.is_leaf():
            curr.accumulate_derivative(scalar_derivatives[curr.unique_id])

    # raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
