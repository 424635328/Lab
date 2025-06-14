# ArmSortApp.pro
QT       += core gui serialport
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG   += c++11
CONFIG   -= app_bundle

DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += \
    main.cpp \
    mainwindow.cpp
HEADERS  += \
    mainwindow.h
FORMS    += \
    mainwindow.ui

# 使用pkg-config自动链接OpenCV
CONFIG += link_pkgconfig
PKGCONFIG += opencv4

# 配置PaddleLite C++预测库
# !!! 关键：请修改为你在实验箱上解压PaddleLite库的实际路径 !!!
PADDLE_LITE_CPP_DIR = /home/linux/paddlelite_cpp/inference_lite_lib.linux.aarch64

INCLUDEPATH += $$PADDLE_LITE_CPP_DIR/include
LIBS += -L$$PADDLE_LITE_CPP_DIR/lib -lpaddle_light_api_shared

# 自动将依赖的动态库拷贝到编译输出目录
PADDLE_LITE_SO = $$PADDLE_LITE_CPP_DIR/lib/libpaddle_light_api_shared.so
DESTDIR_SO = $$OUT_PWD
COPY_CMD = $$QMAKE_COPY $$quote($$PADDLE_LITE_SO) $$quote($$DESTDIR_SO)
QMAKE_POST_LINK += $$COPY_CMD $$escape_expand(\\n\\t)