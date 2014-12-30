#include "camera.h"
#include <mve/camera.h>
#include <new>
#include <Python.h>
#include <structmember.h>
#include "numpy_array.h"

#if PY_MAJOR_VERSION >= 3
#  define PyString_FromString PyUnicode_FromString
#  define PyString_FromFormat PyUnicode_FromFormat
#endif

#pragma clang diagnostic ignored "-Wc++11-compat-deprecated-writable-strings"
#pragma GCC diagnostic ignored "-Wwrite-strings"

/***************************************************************************
 * Camera Info Object
 *
 */

struct CameraInfoObj {
  PyObject_HEAD
  mve::CameraInfo instance;
};

static PyObject* CameraInfo_GetPosition(CameraInfoObj* self, void* closure)
{
  npy_intp dims[] = { 3 };
  PyObject* arr = PyArray_SimpleNew(1, dims, NPY_FLOAT32);

  float* data = (float*) PyArray_DATA((PyArrayObject*)arr);
  self->instance.fill_camera_pos(data);

  return arr;
}

static PyObject* CameraInfo_GetTranslation(CameraInfoObj* self, void* closure)
{
  npy_intp dims[] = { 3 };
  PyObject* arr = PyArray_SimpleNew(1, dims, NPY_FLOAT32);

  float* data = (float*) PyArray_DATA((PyArrayObject*)arr);
  self->instance.fill_camera_translation(data);

  return arr;
}

static PyObject* CameraInfo_GetViewDirection(CameraInfoObj* self, void* closure)
{
  npy_intp dims[] = { 3 };
  PyObject* arr = PyArray_SimpleNew(1, dims, NPY_FLOAT32);

  float* data = (float*) PyArray_DATA((PyArrayObject*)arr);
  self->instance.fill_viewing_direction(data);

  return arr;
}

static PyObject* CameraInfo_GetWorldToCamMatrix(CameraInfoObj* self, void* closure)
{
  npy_intp dims[] = { 4, 4 };
  PyObject* arr = PyArray_SimpleNew(2, dims, NPY_FLOAT32);

  float* data = (float*) PyArray_DATA((PyArrayObject*)arr);
  self->instance.fill_world_to_cam(data);

  return arr;
}

static PyObject* CameraInfo_GetCamToWorldMatrix(CameraInfoObj* self, void* closure)
{
  npy_intp dims[] = { 4, 4 };
  PyObject* arr = PyArray_SimpleNew(2, dims, NPY_FLOAT32);

  float* data = (float*) PyArray_DATA((PyArrayObject*)arr);
  self->instance.fill_cam_to_world(data);

  return arr;
}

static PyObject* CameraInfo_GetWorldToCamRotation(CameraInfoObj* self, void* closure)
{
  npy_intp dims[] = { 3, 3 };
  PyObject* arr = PyArray_SimpleNew(2, dims, NPY_FLOAT32);

  float* data = (float*) PyArray_DATA((PyArrayObject*)arr);
  self->instance.fill_world_to_cam_rot(data);

  return arr;
}

static PyObject* CameraInfo_GetCamToWorldRotation(CameraInfoObj* self, void* closure)
{
  npy_intp dims[] = { 3, 3 };
  PyObject* arr = PyArray_SimpleNew(2, dims, NPY_FLOAT32);

  float* data = (float*) PyArray_DATA((PyArrayObject*)arr);
  self->instance.fill_cam_to_world_rot(data);

  return arr;
}

static PyObject* CameraInfo_GetCalibration(CameraInfoObj* self, PyObject* args, PyObject* kwds)
{
  char* klist[] = { "width", "height", NULL };
  float width, height;

  if (PyArg_ParseTupleAndKeywords(args, kwds, "ff:get_calibration", klist, &width, &height)) {
    npy_intp dims[] = { 3, 3 };
    PyObject* arr = PyArray_SimpleNew(2, dims, NPY_FLOAT32);

    float* data = (float*) PyArray_DATA((PyArrayObject*)arr);
    self->instance.fill_calibration(data, width, height);

    return arr;
  }

  return NULL;
}

static char* CameraInfo_GetCalibration_doc =
"Stores the 3x3 calibration (or projection) matrix (K-matrix in Hartley, Zisserman)\n"
"The matrix projects a point in camera coordinates to the image plane with dimensions \'width\' and \'height\'\n"
"The convention is that the camera looks along the positive z-axis.\n"
"To obtain the pixel coordinates after projection, 0.5 must be subtracted from the coordinates.\n"
"\n"
"If the dimensions of the image are unknown, the generic projection matrix with w=1 and h=1 can be used.\n"
;

//static PyObject* CameraInfo_GetInverseCalibration(CameraInfoObj* self, PyObject* args);
//static PyObject* CameraInfo_GetReprojection

//static PyObject* CameraInfo_GetExtrinsicString(CameraInfoObj* self);
//static PyObject* CameraInfo_GetIntrinsicString(CameraInfoObj* self);

static PyObject* CameraInfo_GetFocalLength(CameraInfoObj* self, void* closure)
{
  return PyFloat_FromDouble(self->instance.flen);
}

static PyObject* CameraInfo_GetPrincipalPoint(CameraInfoObj* self, void* closure)
{
  npy_intp dims[] = { 2 };
  PyObject* arr = PyArray_SimpleNew(1, dims, NPY_FLOAT32);

  float* data = (float*) PyArray_DATA((PyArrayObject*)arr);
  memcpy(data, self->instance.ppoint, 2*sizeof(float));

  return arr;
}

static PyObject* CameraInfo_GetPixelAspect(CameraInfoObj* self, void* closure)
{
  return PyFloat_FromDouble(self->instance.paspect);
}

static PyObject* CameraInfo_GetDistortion(CameraInfoObj* self, void* closure)
{
  npy_intp dims[] = { 2 };
  PyObject* arr = PyArray_SimpleNew(1, dims, NPY_FLOAT32);

  float* data = (float*) PyArray_DATA((PyArrayObject*)arr);
  memcpy(data, self->instance.dist, 2*sizeof(float));

  return arr;
}

static PyMethodDef CameraInfo_methods[] = {
  {"get_calibration", (PyCFunction)CameraInfo_GetCalibration, METH_VARARGS|METH_KEYWORDS, CameraInfo_GetCalibration_doc},
  //{"get_reprojection", },
  //{"get_inverse_calibration"},
  {NULL, NULL, 0, NULL}
};

static PyGetSetDef CameraInfo_getset[] = {
  {"position", (getter)CameraInfo_GetPosition, NULL, "Position", NULL},
  {"translation", (getter)CameraInfo_GetTranslation, NULL, "Translation Vector", NULL},
  {"view_dir", (getter)CameraInfo_GetViewDirection, NULL, "View Direction", NULL},
  {"world_to_cam_matrix", (getter)CameraInfo_GetWorldToCamMatrix, NULL, "World to Camera Matrix", NULL},
  {"cam_to_world_matrix", (getter)CameraInfo_GetCamToWorldMatrix, NULL, "Camera to World Matrix", NULL},
  {"world_to_cam_rotation", (getter)CameraInfo_GetWorldToCamRotation, NULL, "World to Camera Matrix", NULL},
  {"cam_to_world_rotation", (getter)CameraInfo_GetCamToWorldRotation, NULL, "Camera to World Matrix", NULL},
  //{"extrinsic_str"}
  //{"intrinsic_str"}
  {"focal_length", (getter)CameraInfo_GetFocalLength, NULL, "Focal Length", NULL},
  {"principal_point", (getter)CameraInfo_GetPrincipalPoint, NULL, "Principal Point", NULL},
  {"pixel_aspect", (getter)CameraInfo_GetPixelAspect, NULL, "Pixel Aspect", NULL},
  {"distortion", (getter)CameraInfo_GetDistortion, NULL, "Distortion", NULL},
  {NULL, NULL, NULL, NULL, NULL}
};

static PyObject* CameraInfo_New(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
  CameraInfoObj* self = (CameraInfoObj*) subtype->tp_alloc(subtype, 0);

  if (self != NULL) {
    ::new(&(self->instance)) mve::CameraInfo();
  }

  return (PyObject*) self;
}

static void CameraInfo_Dealloc(CameraInfoObj *self)
{
  Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyTypeObject CameraInfoType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "mve.core.CameraInfo", // tp_name
  sizeof(CameraInfoObj), // tp_basicsize
  0, // tp_itemsize
  (destructor)CameraInfo_Dealloc, // tp_dealloc
  0, // tp_print
  0, // tp_getattr (deprecated)
  0, // tp_setattr (deprecated)
#if PY_MAJOR_VERSION < 3
  0, // tp_compare
#else
  0, // reserved
#endif
  0, // tp_repr
  0, // tp_as_number
  0, // tp_as_sequence
  0, // tp_as_mapping
  0, // tp_hash
  0, // tp_call
  0, // tp_str
  0, // tp_getattro
  0, // tp_setattro
  0, // tp_as_buffer
  Py_TPFLAGS_DEFAULT, // tp_flags
  "MVE CameraInfo", // tp_doc
  0, // tp_traverse
  0, // tp_clear
  0, // tp_richcompare
  0, // tp_weaklistoffset
  0, // tp_iter
  0, // tp_iternext
  CameraInfo_methods, // tp_methods
  0, // tp_members
  CameraInfo_getset, // tp_getset
  0, // tp_base
  0, // tp_dict
  0, // tp_descr_get
  0, // tp_descr_set
  0, // tp_dictoffset
  (initproc)0, // tp_init
  0, // tp_alloc
  (newfunc)CameraInfo_New, // tp_new
  0, // tp_free
  0, // tp_is_gc
};

/***************************************************************************
 *
 *
 */

PyObject* CameraInfoObj_Create(const mve::CameraInfo& cam)
{
  PyObject* args = PyTuple_New(0);
  PyObject* kwds = PyDict_New();
  PyObject* obj = CameraInfoType.tp_new(&CameraInfoType, args, kwds);
  Py_DECREF(args);
  Py_DECREF(kwds);

  if (obj) {
    ((CameraInfoObj*) obj)->instance = cam;
  }

  return obj;
}

mve::CameraInfo& CameraInfo_AsMveCameraInfo(PyObject* self)
{
  return ((CameraInfoObj*) self)->instance;
}

void load_Camera(PyObject* mod)
{
  if (PyType_Ready(&CameraInfoType) < 0)
    abort();
  Py_INCREF(&CameraInfoType);

  PyModule_AddObject(mod, "CameraInfo", (PyObject*)&CameraInfoType);
}
