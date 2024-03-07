import os

def do_log(rank, tag, info=""):
  file_name = "/Megatron-LLaMA/examples_of_zhang/log/log_"+str(rank)+".txt"
  if os.path.exists(file_name):
    with open(file_name, "a") as log_file:
      log_file.write(tag+" "+str(info)+"\n")
  else:
    with open(file_name, "w") as log_file:
      log_file.write(tag+" "+str(info)+"!\n")