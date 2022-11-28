from z3 import * 

ego_x = Real('ego_x')
ego_y = Real('ego_y')
ego_vx = Real('ego_vx')
ego_vy = Real('ego_vy')

output_x = Real('output_x')
output_y = Real('output_y')
output_vx = Real('output_vx')
output_vy = Real('output_vy')

# y_solver_list = []
# tmp = Solver()
# tmp.add(output_y==20)
# tmp.add(ego_y>20)
# y_solver_list.append(tmp)
# tmp = Solver()
# tmp.add(ego_y<0)
# tmp.add(output_y==0)
# y_solver_list.append(tmp)

# vy_solver_list = []
# tmp = Solver()
# tmp.add(output_vy==-ego_vy)
# tmp.add(ego_y>20)
# vy_solver_list.append(tmp)
# tmp = Solver()
# tmp.add(ego_y<0)
# tmp.add(output_vy==-ego_vy)
# vy_solver_list.append(tmp)

# x_solver_list = []
# tmp = Solver()
# tmp.add(output_x==20)
# tmp.add(ego_x>20)
# x_solver_list.append(tmp)
# tmp = Solver()
# tmp.add(ego_x<0)
# tmp.add(output_x==0)
# x_solver_list.append(tmp)

# vx_solver_list = []
# tmp = Solver()
# tmp.add(output_vx==-ego_vx)
# tmp.add(ego_x>20)
# vx_solver_list.append(tmp)
# tmp = Solver()
# tmp.add(ego_x<0)
# tmp.add(output_vx==-ego_vx)
# vx_solver_list.append(tmp)

s = Solver()
cst1 =And(ego_y>20, output_vy==-ego_vy, output_y==20) 
cst2 =And(ego_x>20, output_vx==-ego_vx, output_x==20) 
cst3 =And(ego_y<0, output_vy==-ego_vy, output_y==0) 
cst4 =And(ego_x<0, output_vx==-ego_vx, output_x==0) 
s.add(Or(cst1, cst2, cst3, cst4))
s.push()

s.add(ego_x>19)
s.add(ego_x<21)
s.add(ego_y>4)
s.add(ego_y<5)
s.add(ego_vx==10)
s.add(ego_vy==10)

print(s.check())
m = s.model()
print(m)

