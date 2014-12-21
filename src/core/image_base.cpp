#include "image_base.h"
#include <mve/image_base.h>
#include <mve/image.h>
#include <Python.h>
#include "numpy_array.h"

#if PY_MAJOR_VERSION >= 3
#  define PyString_FromFormat PyUnicode_FromFormat
#endif

#pragma GCC diagnostic ignored "-Wc++11-compat-deprecated-writable-strings"

static int _ImageTypeToNumpyDataType(mve::ImageType ty)
{
  switch (ty) {
    case mve::IMAGE_TYPE_UINT8: return NPY_UINT8;
    case mve::IMAGE_TYPE_UINT16: return NPY_UINT16;
    case mve::IMAGE_TYPE_UINT32: return NPY_UINT32;
    case mve::IMAGE_TYPE_UINT64: return NPY_UINT64;
    case mve::IMAGE_TYPE_SINT8: return NPY_INT8;
    case mve::IMAGE_TYPE_SINT16: return NPY_INT16;
    case mve::IMAGE_TYPE_SINT32: return NPY_INT32;
    case mve::IMAGE_TYPE_SINT64: return NPY_INT64;
    case mve::IMAGE_TYPE_FLOAT: return NPY_FLOAT32;
    case mve::IMAGE_TYPE_DOUBLE: return NPY_FLOAT64;
    case mve::IMAGE_TYPE_UNKNOWN: return NPY_NOTYPE;
  };
  return NPY_NOTYPE;
}

/***************************************************************************
 * ImageBase Object
 *
 */

struct ImageBaseObj {
  PyObject_HEAD
  mve::ImageBase::Ptr thisptr;
};

static PyObject* ImageBase_Clone(ImageBaseObj *self)
{
  mve::ImageBase::Ptr ptr = self->thisptr->duplicate();
  return ImageBase_Create(ptr);
}

static PyMethodDef ImageBase_methods[] = {
  {"clone", (PyCFunction)ImageBase_Clone, METH_NOARGS, "Clone"},
  {NULL, NULL, 0, NULL}
};

static PyObject* ImageBase_GetWidth(ImageBaseObj *self, void* closure)
{
  return PyLong_FromLong(self->thisptr->width());
}

static PyObject* ImageBase_GetHeight(ImageBaseObj *self, void* closure)
{
  return PyLong_FromLong(self->thisptr->height());
}

static PyObject* ImageBase_GetChannels(ImageBaseObj *self, void* closure)
{
  return PyLong_FromLong(self->thisptr->channels());
}

static PyObject* ImageBase_GetByteSize(ImageBaseObj *self, void* closure)
{
  return PyLong_FromLong(self->thisptr->get_byte_size());
}

static PyObject* ImageBase_GetImageType(ImageBaseObj *self, void* closure)
{
  return PyLong_FromLong(self->thisptr->get_type());
}

static PyObject* ImageBase_GetData(ImageBaseObj *self, void* closure)
{
  mve::ImageBase::Ptr ptr = self->thisptr;

  int ndim = (ptr->channels() == 1 ? 2 : 3);
  npy_intp dims[] = { ptr->height(), ptr->width(), ptr->channels() };
  int dtype = _ImageTypeToNumpyDataType(ptr->get_type());
  void *data = ptr->get_byte_pointer();

  PyObject* arr = PyArray_SimpleNewFromData(ndim, dims, dtype, data);
  if (!arr)
    return NULL;

  Py_INCREF((PyObject*) self);
  PyArray_SetBaseObject((PyArrayObject*)arr, (PyObject*)self); // steal

  return arr;
}

static PyGetSetDef ImageBase_getset[] = {
  {"width", (getter)ImageBase_GetWidth, 0, "Width", NULL },
  {"height", (getter)ImageBase_GetHeight, 0, "Height", NULL},
  {"channels", (getter)ImageBase_GetChannels, 0, "Channels", NULL},
  {"byte_size", (getter)ImageBase_GetByteSize, 0, "Size in Bytes", NULL},
  {"image_type", (getter)ImageBase_GetImageType, 0, "Image Type", NULL},
  {"data", (getter)ImageBase_GetData, 0, "Data", NULL},
  {NULL, NULL, NULL, NULL, NULL}
};

static int ImageBase_Init(ImageBaseObj *self, PyObject *args, PyObject *kwds)
{
  char* klist[] = { "width", "height", "channels", "type", NULL };
  int width, height, channels, type;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiii:init", klist,
                                   &width, &height, &channels, &type))
    return -1;

  //self->thisptr = new mve::ImageBase();

  switch (type) {
    case mve::IMAGE_TYPE_UINT8:
      self->thisptr = mve::Image<uint8_t>::create(width, height, channels);
      break;
    case mve::IMAGE_TYPE_UINT16:
      self->thisptr = mve::Image<uint16_t>::create(width, height, channels);
      break;
    case mve::IMAGE_TYPE_UINT32:
      self->thisptr = mve::Image<uint32_t>::create(width, height, channels);
      break;
    case mve::IMAGE_TYPE_UINT64:
      self->thisptr = mve::Image<uint64_t>::create(width, height, channels);
      break;
    case mve::IMAGE_TYPE_SINT8:
      self->thisptr = mve::Image<int8_t>::create(width, height, channels);
      break;
    case mve::IMAGE_TYPE_SINT16:
      self->thisptr = mve::Image<int16_t>::create(width, height, channels);
      break;
    case mve::IMAGE_TYPE_SINT32:
      self->thisptr = mve::Image<int32_t>::create(width, height, channels);
      break;
    case mve::IMAGE_TYPE_SINT64:
      self->thisptr = mve::Image<int64_t>::create(width, height, channels);
      break;
    case mve::IMAGE_TYPE_FLOAT:
      self->thisptr = mve::Image<float>::create(width, height, channels);
      break;
    case mve::IMAGE_TYPE_DOUBLE:
      self->thisptr = mve::Image<double>::create(width, height, channels);
      break;
    default:
      PyErr_SetString(PyExc_TypeError, "Invalid Image Type");
      return -1;
  }

  return 0;
}

static PyObject* ImageBase_New(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
  ImageBaseObj* self = (ImageBaseObj*) subtype->tp_alloc(subtype, 0);

  if (self != NULL) {
    ::new(&(self->thisptr)) mve::ImageBase::Ptr();
  }

  return (PyObject*) self;
}

static void ImageBase_Dealloc(ImageBaseObj *self)
{
  //printf("image is deallocated\n");
  self->thisptr.reset();
  Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyObject* ImageBase_Representation(ImageBaseObj *self)
{
  return PyString_FromFormat("ImageBase(%d x %d x %s[%d])",
                             self->thisptr->width(),
                             self->thisptr->height(),
                             self->thisptr->get_type_string(),
                             self->thisptr->channels());
}

static PyTypeObject ImageBaseType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "mve.core.ImageBase", // tp_name
  sizeof(ImageBaseObj), // tp_basicsize
  0, // tp_itemsize
  (destructor)ImageBase_Dealloc, // tp_dealloc
  0, // tp_print
  0, // tp_getattr (deprecated)
  0, // tp_setattr (deprecated)
#if PY_MAJOR_VERSION < 3
  0, // tp_compare
#else
  0, // reserved
#endif
  (reprfunc)ImageBase_Representation, // tp_repr
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
  "MVE ImageBase", // tp_doc
  0, // tp_traverse
  0, // tp_clear
  0, // tp_richcompare
  0, // tp_weaklistoffset
  0, // tp_iter
  0, // tp_iternext
  ImageBase_methods, // tp_methods
  0, // tp_members
  ImageBase_getset, // tp_getset
  0, // tp_base
  0, // tp_dict
  0, // tp_descr_get
  0, // tp_descr_set
  0, // tp_dictoffset
  (initproc)ImageBase_Init, // tp_init
  0, // tp_alloc
  (newfunc)ImageBase_New, // tp_new
  0, // tp_free
  0, // tp_is_gc
 };

/***************************************************************************
 *
 *
 */

PyObject* ImageBase_Create(mve::ImageBase::Ptr ptr)
{
  if (ptr.get() == NULL) {
    abort();
  }

  PyObject* args = PyTuple_New(0);
  PyObject* kwds = PyDict_New();
  PyObject* obj = ImageBaseType.tp_new(&ImageBaseType, args, kwds);
  Py_DECREF(args);
  Py_DECREF(kwds);

  if (obj) {
    ((ImageBaseObj*) obj)->thisptr = ptr;
  }

  return obj;
}

bool ImageBase_Check(PyObject* obj)
{
  return PyObject_IsInstance(obj, (PyObject*) &ImageBaseType);
}

mve::ImageBase::Ptr ImageBase_GetImagePtr(PyObject* obj)
{
  if (ImageBase_Check(obj)) {
    return ((ImageBaseObj*) obj)->thisptr;
  }
  return mve::ImageBase::Ptr();
}

void load_ImageBase(PyObject *mod)
{
  if (PyType_Ready(&ImageBaseType) < 0)
    abort();
  Py_INCREF(&ImageBaseType);

  PyModule_AddObject(mod, "ImageBase", (PyObject*)&ImageBaseType);

  PyModule_AddIntConstant(mod, "IMAGE_TYPE_UNKNOWN", mve::IMAGE_TYPE_UNKNOWN);
  PyModule_AddIntConstant(mod, "IMAGE_TYPE_UINT8", mve::IMAGE_TYPE_UINT8);
  PyModule_AddIntConstant(mod, "IMAGE_TYPE_UINT16", mve::IMAGE_TYPE_UINT16);
  PyModule_AddIntConstant(mod, "IMAGE_TYPE_UINT32", mve::IMAGE_TYPE_UINT32);
  PyModule_AddIntConstant(mod, "IMAGE_TYPE_UINT64", mve::IMAGE_TYPE_UINT64);
  PyModule_AddIntConstant(mod, "IMAGE_TYPE_SINT8", mve::IMAGE_TYPE_SINT8);
  PyModule_AddIntConstant(mod, "IMAGE_TYPE_SINT16", mve::IMAGE_TYPE_SINT16);
  PyModule_AddIntConstant(mod, "IMAGE_TYPE_SINT32", mve::IMAGE_TYPE_SINT32);
  PyModule_AddIntConstant(mod, "IMAGE_TYPE_SINT64", mve::IMAGE_TYPE_SINT64);
  PyModule_AddIntConstant(mod, "IMAGE_TYPE_FLOAT", mve::IMAGE_TYPE_FLOAT);
  PyModule_AddIntConstant(mod, "IMAGE_TYPE_DOUBLE", mve::IMAGE_TYPE_DOUBLE);
}
