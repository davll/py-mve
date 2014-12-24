#pragma once

#include <mve/image_base.h>
#include <Python.h>

PyObject* Image_Create(mve::ImageBase::Ptr ptr);

bool Image_Check(PyObject* obj);

PyTypeObject* Image_Type();

mve::ImageBase::Ptr Image_GetImageBasePtr(PyObject* obj);

void load_Image(PyObject *mod);
