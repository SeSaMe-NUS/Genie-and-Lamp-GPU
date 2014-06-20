#!/usr/bin/ruby -W0

DEBUG = false

def cases
  [
  #[topk, dimension, query]
    [2,   8,         2],
    [4,   16,        8],
    [8,   32,        32],
    [16,  64,        128],
    [32,  64,        128],
    [64,  128,       128],
    [128, 128,       256],
    [128, 256,       1028],
  ]
end

def release
  "Release/Genie-and-Lamp-GPU"
end

def debug
  "Debug/Genie-and-Lamp-GPU"
end

# MAIN func starts
class String
  def red;            "\033[31m#{self}\033[0m" end
end

script = DEBUG ? debug : release

for arg in cases do
  runner = "#{script} -k #{arg[0]} -d #{arg[1]} -q #{arg[2]}"
  puts "==> running #{runner}".red
  result = `#{runner}`
  gpu_time =  /finished\s+with\s+total\s+time\s+\:\s+\d+\.\d+\s+with\s+\d+\s+iterations/.match(result) || /finished\s+with\s+total\s+time\s+\:\s+\d+\s+with\s+\d+\s+iterations/.match(result)
  cpu_scan_time = /the\s+time\s+of\s+top\-\d+\s+in\s+CPU\s+version\s+is\:\d+\.\d+/.match(result) || /the\s+time\s+of\s+top\-\d+\s+in\s+CPU\s+version\s+is\:\d+/.match(result)
  gpu_scan_time = ""
  puts "TOPK = #{arg[0]} DIMENSIONNUM = #{arg[1]} QUERYNUM = #{arg[2]}"
  puts "GPU     : " + gpu_time[0]
  puts "CPU_SCAN: " + cpu_scan_time[0]
  puts "GPU_SCAN: " + gpu_scan_time

  puts
end
