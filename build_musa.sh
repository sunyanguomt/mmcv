export CFLAGS="-g -O0"
export CPPFLAGS="-DDEBUG=1"

DEBUG=1 MMCV_WITH_OPS=1 MUSA_ARCH=22 FORCE_MUSA=1 python setup.py install
