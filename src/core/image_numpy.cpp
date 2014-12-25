#include "image_numpy.h"
#include <Python.h>
#include "numpy_array.h"

int _ImageTypeToNumpyDataType(mve::ImageType ty)
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

mve::ImageType _NumpyDataTypeToImageType(int dtype)
{
  switch (dtype) {
    case NPY_UINT8: return mve::IMAGE_TYPE_UINT8;
    case NPY_UINT16: return mve::IMAGE_TYPE_UINT16;
    case NPY_UINT32: return mve::IMAGE_TYPE_UINT32;
    case NPY_UINT64: return mve::IMAGE_TYPE_UINT64;
    case NPY_INT8: return mve::IMAGE_TYPE_SINT8;
    case NPY_INT16: return mve::IMAGE_TYPE_SINT16;
    case NPY_INT32: return mve::IMAGE_TYPE_SINT32;
    case NPY_INT64: return mve::IMAGE_TYPE_SINT64;
    case NPY_FLOAT32: return mve::IMAGE_TYPE_FLOAT;
    case NPY_FLOAT64: return mve::IMAGE_TYPE_DOUBLE;
  };
  return mve::IMAGE_TYPE_UNKNOWN;
}

size_t _ImageTypeElementSize(mve::ImageType ty)
{
  switch (ty) {
    case mve::IMAGE_TYPE_UINT8: return sizeof(uint8_t);
    case mve::IMAGE_TYPE_UINT16: return sizeof(uint16_t);
    case mve::IMAGE_TYPE_UINT32: return sizeof(uint32_t);
    case mve::IMAGE_TYPE_UINT64: return sizeof(uint64_t);
    case mve::IMAGE_TYPE_SINT8: return sizeof(int8_t);
    case mve::IMAGE_TYPE_SINT16: return sizeof(int16_t);
    case mve::IMAGE_TYPE_SINT32: return sizeof(int32_t);
    case mve::IMAGE_TYPE_SINT64: return sizeof(int64_t);
    case mve::IMAGE_TYPE_FLOAT: return sizeof(float);
    case mve::IMAGE_TYPE_DOUBLE: return sizeof(double);
    case mve::IMAGE_TYPE_UNKNOWN: return 0;
  };
  return (size_t)-1;
}

char const * _ImageTypeToString(mve::ImageType ty)
{
  switch (ty) {
    case mve::IMAGE_TYPE_UINT8: return "uint8";
    case mve::IMAGE_TYPE_UINT16: return "uint16";
    case mve::IMAGE_TYPE_UINT32: return "uint32";
    case mve::IMAGE_TYPE_UINT64: return "uint64";
    case mve::IMAGE_TYPE_SINT8: return "sint8";
    case mve::IMAGE_TYPE_SINT16: return "sint16";
    case mve::IMAGE_TYPE_SINT32: return "sint32";
    case mve::IMAGE_TYPE_SINT64: return "sint64";
    case mve::IMAGE_TYPE_FLOAT: return "float";
    case mve::IMAGE_TYPE_DOUBLE: return "double";
    case mve::IMAGE_TYPE_UNKNOWN: return "unknown";
  };
  return "unknown";
}

PyArrayImage::PyArrayImage()
: ImageBase(), array(NULL)
{
}

PyArrayImage::~PyArrayImage()
{
  Py_XDECREF(array);
}

mve::ImageBase::Ptr PyArrayImage::duplicate() const
{
  PyArrayImage *ptr = new PyArrayImage();
  mve::ImageBase::Ptr ret_ptr(ptr);

  PyArrayObject *arr = NULL, *old_arr = (PyArrayObject*) this->array;

  arr = (PyArrayObject*)PyArray_NewLikeArray(old_arr, NPY_CORDER, NULL, NPY_TRUE);
  if (!arr)
    return mve::ImageBase::Ptr();

  if (PyArray_CopyInto(arr, old_arr) < 0) {
    Py_DECREF(arr);
    return mve::ImageBase::Ptr();
  }

  int ndim = PyArray_NDIM(arr);
  ptr->array = (PyObject*) arr;
  ptr->type = this->type;
  ptr->w = PyArray_DIM(arr, 1);
  ptr->h = PyArray_DIM(arr, 0);
  ptr->c = (ndim == 2 ? 1 : PyArray_DIM(arr, 2));
  return ret_ptr;
}

int PyArrayImage::copy_from(PyObject *obj)
{
  Py_XDECREF(array);

  PyArrayObject *arr = NULL, *old_arr = (PyArrayObject*)obj;

  if (!PyArray_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "Argument should be a Numpy Array");
    return -1;
  }

  int ndim = PyArray_NDIM(old_arr);
  if (ndim != 2 && ndim != 3) {
    PyErr_SetString(PyExc_TypeError, "Argument should be a 2/3 dimensional array");
    return -1;
  }

  mve::ImageType ty = _NumpyDataTypeToImageType(PyArray_TYPE(old_arr));
  if (ty == mve::IMAGE_TYPE_UNKNOWN) {
    PyErr_SetString(PyExc_TypeError, "Argument should be a array with known scalar type");
    return -1;
  }

  arr = (PyArrayObject*)PyArray_NewLikeArray(old_arr, NPY_CORDER, NULL, NPY_TRUE);
  if (!arr) {
    return -1;
  }

  if (PyArray_CopyInto(arr, (PyArrayObject*)obj) < 0) {
    Py_DECREF(arr);
    return -1;
  }

  this->array = (PyObject*) arr;
  this->type = ty;
  this->w = PyArray_DIM(arr, 1);
  this->h = PyArray_DIM(arr, 0);
  this->c = (ndim == 2 ? 1 : PyArray_DIM(arr, 2));
  return 0;
}

std::size_t PyArrayImage::get_byte_size() const
{
  if (!array)
    return 0;
  size_t elemsize = _ImageTypeElementSize(get_type());
  return PyArray_SIZE((PyArrayObject*)array) * elemsize;
}

char const* PyArrayImage::get_byte_pointer() const
{
  if (!array)
    return NULL;
  return PyArray_BYTES((PyArrayObject*)array);
}

char * PyArrayImage::get_byte_pointer()
{
  if (!array)
    return NULL;
  return PyArray_BYTES((PyArrayObject*)array);
}

mve::ImageType PyArrayImage::get_type() const
{
  if (!array)
    return mve::IMAGE_TYPE_UNKNOWN;
  return type;
}

char const* PyArrayImage::get_type_string() const
{
  return _ImageTypeToString(get_type());
}
