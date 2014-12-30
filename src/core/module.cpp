#include <Python.h>
#define MODULE_STR "mve.core"
#define MODULE_NAME core
#include "scene.h"
#include "view.h"
#include "camera.h"
#include "image.h"

#define IMPORT_ARRAY
#include "numpy_array.h"

#pragma clang diagnostic ignored "-Wc++11-compat-deprecated-writable-strings"
#pragma GCC diagnostic ignored "-Wwrite-strings"

static PyMethodDef module_methods[] = {
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef mve_moduledef = {
  PyModuleDef_HEAD_INIT,
  MODULE_STR,
  "Python wrapper for Multi-View Environment",
  -1,
  module_methods
};

PyMODINIT_FUNC PyInit_core()
#else
PyMODINIT_FUNC initcore()
#endif
{
  // Create Core Module
#if PY_MAJOR_VERSION >= 3
  PyObject* mod = PyModule_Create(&mve_moduledef);
#else
  PyObject* mod = Py_InitModule(MODULE_STR, module_methods);
#endif

  // Import Numpy Array API
  import_array();

  // Check Numpy Version
  if (NPY_VERSION != PyArray_GetNDArrayCVersion()) {
    // TODO: raise API incompatibility error
  }
  if (NPY_FEATURE_VERSION > PyArray_GetNDArrayCFeatureVersion()) {
    // TODO: raise API incompatibility error
  }

  // Load Class
  load_Scene(mod);
  load_View(mod);
  load_Camera(mod);
  load_Image(mod);

#if PY_MAJOR_VERSION >= 3
  return mod;
#endif
}
