cp -r ../yukarin_wavernn/src_cython ./yukarin_autoreg_cython
cp ../yukarin_autoreg_cpp/yukarin_autoreg_cpp.lib ./yukarin_autoreg_cython/yukarin_autoreg_cpp.lib
cp ../yukarin_autoreg_cpp/yukarin_autoreg_cpp.dll ./yukarin_autoreg_cython/yukarin_autoreg_cpp.dll

cd ./yukarin_autoreg_cython

python setup.py clean
python setup.py install
rm yukarin_autoreg_cpp.cpp; rm -r build

# python check.py

cd ../
cp ./yukarin_autoreg_cython/yukarin_autoreg_cpp.dll ./
