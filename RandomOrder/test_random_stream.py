import os
import sys
import signal
import time
import subprocess

def signal_handler(signal,frame):
  print("bye")
  sys.exit(0)
signal.signal(signal.SIGINT,signal_handler)

def shellGetOutput(str) :
  process = subprocess.Popen(str,shell=True,stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
  output, err = process.communicate()

  if (len(err) > 0):
    raise NameError(str+"\n"+output.decode('utf-8')+err.decode('utf-8'))
  return output.decode('utf-8')

def appendToFile(out, filename):
  with open(filename, "a+") as out_file:
    out_file.writelines(out)

def main():
  # Read parameters from setup file
  with open('setup.txt') as parameters_file:
    for line in parameters_file:
      line = line.strip()
      split = [x.strip() for x in line.split(':')]
      if len(split) <= 1:
        continue
      params = [x.strip() for x in split[1].split(',')]
      if line.startswith("Input graph directory"):
        read_dir = split[1]
      elif line.startswith("Dynamic graphs"):
        files = params.copy()
      elif line.startswith("Output directory"):
        write_dir = split[1]
      elif line.startswith("Benchmarks"):
        programs = params
      elif line.startswith("Numbers of workers"):
        num_workers = params.copy()
      elif line.startswith("Percent"):
        percents = params.copy()
  # Setup other parameters
  program_dir = "./gbbs/benchmarks/StreamingTriangles/"
  empty = "gbbs/benchmarks/EdgeOrientation/ParallelLDS/empty_h"
  # for program in programs:
  #  program_path = os.path.join(program_dir, program)
  #  program_local_dir = os.path.dirname(program_path)
  #  subprocess.Popen(["make"], stdout=subprocess.PIPE, cwd=program_local_dir)

  git_init_process = subprocess.Popen("git init gbbs/",shell=True,stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
  output, err = git_init_process.communicate()

  for file_idx, filename in enumerate(files):
    for program_idx, program in enumerate(programs):
      for p in percents:
        num_rounds = 1
        out_path_components = [programs[program_idx], filename, p, ".out"]
        out_filename = os.path.join(write_dir, "_".join(out_path_components))
        program_path = os.path.join(program_dir, program)
        ss = (program_path + " -i " + read_dir + filename + " -s -rand -rand_trial -percent " + p + " "
                "-rounds 1 " + empty)
        print(ss)
        out = shellGetOutput(ss)
        appendToFile(out, out_filename)

if __name__ == "__main__":
  main()
