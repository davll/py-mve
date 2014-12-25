#pragma once
#include <mve/image_base.h>
#include <Python.h>

class PyArrayImage : public mve::ImageBase {
public:
  PyArrayImage (void);
  virtual ~PyArrayImage (void);

  /** Copy content from PyArray object */
  int copy_from(PyObject *obj);

  /** Duplicates the image. Data holders need to reimplement this. */
  virtual mve::ImageBase::Ptr duplicate (void) const;

  /** Generic byte size information. Returns 0 if not overwritten. */
  virtual std::size_t get_byte_size (void) const;
  /** Pointer to image data. Returns 0 if not overwritten. */
  virtual char const* get_byte_pointer (void) const;
  /** Pointer to image data. Returns 0 if not overwritten. */
  virtual char* get_byte_pointer (void);
  /** Value type information. Returns UNKNOWN if not overwritten. */
  virtual mve::ImageType get_type (void) const;
  /** Returns a string representation of the image data type. */
  virtual char const* get_type_string (void) const;

private:
  PyObject *array;
  mve::ImageType type;
};

int _ImageTypeToNumpyDataType(mve::ImageType ty);
mve::ImageType _NumpyDataTypeToImageType(int dtype);
size_t _ImageTypeElementSize(mve::ImageType ty);
char const * _ImageTypeToString(mve::ImageType ty);
