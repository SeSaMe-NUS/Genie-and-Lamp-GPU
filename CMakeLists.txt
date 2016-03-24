cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

set(APPLICATION_EXECUTABLE "Genie-and-Lamp-GPU")
set(APPLICATION_SOURCES
	src/CONSTANT.cpp
	src/Genie/GPUKNN/GPUManager.cu
	src/Genie/GPUKNN/generalization.cu
	src/Genie/GPUKNN/UtlGPU.cu
	src/Genie/WrapperScan.cpp
	src/Genie/WrapperGPUKNN.cpp
	src/Genie/Scan/GPUScan.cu
	src/Genie/Scan/CPUScan.cu
	src/Genie/Scan/UtlScan.cu
	src/Main.cu
	src/AppManager/deviceDetector/deviceDetector.cu
	src/AppManager/IndexBuilder/DataProcess.cpp
	src/AppManager/IndexBuilder/invListBuilder.cpp
	src/AppManager/WrapperIndexBuilder.cpp
)
cuda_add_executable(${APPLICATION_EXECUTABLE} ${APPLICATION_SOURCES})

