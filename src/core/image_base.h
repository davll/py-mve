#pragma once

#include <mve/image_base.h>
#include <Python.h>

PyObject* ImageBase_Create(mve::ImageBase::Ptr ptr);

bool ImageBase_IsImage(PyObject* obj);

mve::ImageBase::Ptr ImageBase_GetImagePtr(PyObject* obj);

void load_ImageBase(PyObject *mod);
