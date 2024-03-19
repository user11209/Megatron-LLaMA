# from global_variable.global_variable import print_n_increase_global, print_global#, _GLOBAL_VAR

# print_n_increase_global()
# print_n_increase_global()
# print_n_increase_global()
# print_global()



# import global_variable

# global_variable.print_n_increase_global()
# global_variable.print_n_increase_global()
# global_variable.print_n_increase_global()
# global_variable.print_global()



from mid_dir import global_variable

def do_something():
  global_variable.print_n_increase_global()
  global_variable.print_n_increase_global()
  global_variable.print_n_increase_global()
  global_variable.print_global()
  print(global_variable._GLOBAL_VAR)


if __name__=="__main__":
  do_something()