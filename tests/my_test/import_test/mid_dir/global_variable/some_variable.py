_GLOBAL_VAR = None

def print_n_increase_global():
  global _GLOBAL_VAR
  print(_GLOBAL_VAR)
  if _GLOBAL_VAR == None:
    _GLOBAL_VAR = 0
  else:
    _GLOBAL_VAR = _GLOBAL_VAR + 1
  return _GLOBAL_VAR

def print_global():
  global _GLOBAL_VAR
  print(_GLOBAL_VAR)
  return

if __name__ == "__main__":
  print_n_increase_global()
  print_n_increase_global()
  print_n_increase_global()