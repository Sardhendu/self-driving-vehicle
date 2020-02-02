
import numpy as np
np.random.seed(10)
a = np.random.random((10,6))
print(a)

x = np.array([2,2, 2, 2, 3, 3, 2, 2, 3, 3])
y = np.array([0,1,2,3,4,5,6,7,8,9])

a[a<0.2] = 0
a[a>0] = 1

print(a)

x_margin1 = x-1
x_margin2 = x+1

active_idx_y, active_idx_x = np.nonzero(a)

print()
print(active_idx_y)
print(active_idx_x)
# print(repeted_val)
print()
print(x_margin1)
print(x_margin2)

# print(y[active_idx_y])
x_margin1_ext = x_margin1[active_idx_y]
x_margin2_ext = x_margin2[active_idx_y]

print('\n\n')
print(active_idx_x)
print('')
print(x_margin1_ext)
print('')
print(x_margin2_ext)
# new_x = np.arange()

ll_inds = (
        (active_idx_x > x_margin1_ext) & (active_idx_x < x_margin2_ext)
)

print('OUTPUT: ')
x_active = active_idx_x[ll_inds]
y_active = active_idx_y[ll_inds]
print(x_active, y_active)


a[y_active, x_active] = 2

print(a)
