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

def benchmarkToProgramPath(benchmark):
  benchmark_dict = {
    "StreamingTriangle" : "./StreamingTriangle"
  }
  return benchmark_dict.get(benchmark)

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
        programs = [benchmarkToProgramPath(x) for x in params]
        program_pres = params.copy()
      elif line.startswith("Percent"):
        percent = params.copy()
  # Setup other parameters
  for file_idx, filename in enumerate(files):
    for program_idx, program in enumerate(programs):
      for p in percent:
        out_path_components = [program_pres[program_idx], filename, p, ".out"]
        read_filename = os.path.join(write_dir, "_".join(out_path_components))

        with open(read_filename, "r") as read_file:
          for line in read_file:
            line = line.strip()
            if "Mean" in line:
              split = [x.strip() for x in line.split(':')]
              mean = split[1]
            elif "Standard" in line:
              split = [x.strip() for x in line.split(':')]
              dev = split[1]
        print(str(filename), end=",")
        print(str(p), end=",")
        print(str(mean), end=",")
        print(str(dev), end="\n")

if __name__ == "__main__":
  main()
