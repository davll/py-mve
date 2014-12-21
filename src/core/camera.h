#pragma once

#include <mve/camera.h>
#include <Python.h>

PyObject* CameraInfoObj_Create(const mve::CameraInfo&);
mve::CameraInfo& CameraInfo_AsMveCameraInfo(PyObject*);

void load_Camera(PyObject* mod);
