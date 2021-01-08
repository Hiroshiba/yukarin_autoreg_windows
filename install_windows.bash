cp -r ../yukarin_autoreg/src_cython ./yukarin_autoreg_cython
cp ../yukarin_autoreg_cpp/CppWaveRNN/CppWaveRNN.h ./yukarin_autoreg_cython/CppWaveRNN.h
cp ../yukarin_autoreg_cpp/CppWaveRNN/libyukarin_autoreg_cpp.lib ./yukarin_autoreg_cython/yukarin_autoreg_cpp.lib
cp ../yukarin_autoreg_cpp/CppWaveRNN/libyukarin_autoreg_cpp.dll ./yukarin_autoreg_cython/libyukarin_autoreg_cpp.dll

cd ./yukarin_autoreg_cython
python setup.py build_ext --inplace
python setup.py install

cd ../
cp ./yukarin_autoreg_cython/libyukarin_autoreg_cpp.dll ./
