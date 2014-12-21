#pragma once

#include <mve/view.h>
#include <Python.h>

PyObject* ViewObj_Create(mve::View::Ptr ptr);

void load_View(PyObject* mod);
