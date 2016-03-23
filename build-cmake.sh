BUILD=Release  # Can be either "Debug" or "Release"

mkdir -p ${BUILD}
cd ${BUILD}
cmake .. -DCMAKE_BUILD_TYPE=${BUILD}
make -j4
cd ..

