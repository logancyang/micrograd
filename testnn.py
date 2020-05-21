# %%
from micrograd.engine import Value
from micrograd.trace_graph import draw_dot

w = Value(3.0, name='w')
x = Value(-4.0, name='x')
b = Value(2.0, name='b')
y = w*x + b
y.set_name('y')
print("y:", y)
y.backward()

draw_dot(y)
# %%
!pip list




# %%
