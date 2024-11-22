TEMPLATE    = lib
CONFIG     += plugin
QT += widgets
#INCLUDEPATH += ../MainApp
TARGET = OnnxPlugin
DESTDIR = ../plugins
CONFIG(debug, debug|release) {
    TARGET = $$join(TARGET,,,_d)
}

INCLUDEPATH+=$$PWD/include-gpu
INCLUDEPATH += $$PWD/../../CommonComponent/API_Opencv
include ($$PWD/../../CommonComponent/API_Opencv/QtOpencvSet.pri)

LIBS+=-L$$PWD/lib-gpu\
-lonnxruntime\
-lonnxruntime_providers_cuda\
-lonnxruntime_providers_shared\
-lonnxruntime_providers_tensorrt\

HEADERS += \
    onnxplugin.h \
    ../onnxinterface.h

SOURCES += \
    onnxplugin.cpp
