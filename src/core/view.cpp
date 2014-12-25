#include "view.h"
#include "camera.h"
#include "image.h"
#include <mve/view.h>
#include <util/exception.h>
#include <new>
#include <Python.h>
#include <structmember.h>
#include "numpy_array.h"

#if PY_MAJOR_VERSION >= 3
#  define PyString_FromString PyUnicode_FromString
#  define PyString_FromFormat PyUnicode_FromFormat
#endif

#pragma GCC diagnostic ignored "-Wc++11-compat-deprecated-writable-strings"

/***************************************************************************
 * View Object
 *
 */

struct ViewObj {
  PyObject_HEAD
  mve::View::Ptr thisptr;
};

static PyObject* View_CleanupCache(ViewObj *self)
{
  self->thisptr->cache_cleanup();
  Py_RETURN_NONE;
}

static PyObject* View_HasImage(ViewObj *self, PyObject *arg)
{
  if (PyUnicode_Check(arg)) {
    PyObject *bytes = PyUnicode_AsUTF8String(arg);
    PyObject *result = View_HasImage(self, bytes);
    Py_DECREF(bytes);
    return result;
  }

  const char* name = PyBytes_AsString(arg);
  if (!name)
    return NULL;

  if (self->thisptr->has_image_embedding(name)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject* View_GetImage(ViewObj *self, PyObject *arg)
{
  if (PyUnicode_Check(arg)) {
    PyObject *bytes = PyUnicode_AsUTF8String(arg);
    PyObject *result = View_GetImage(self, bytes);
    Py_DECREF(bytes);
    return result;
  }

  const char* name = PyBytes_AsString(arg);
  if (!name)
    return NULL;

  mve::ImageBase::Ptr ptr = self->thisptr->get_image(name);

  if (ptr.get() != NULL)
    return Image_Create(ptr);

  Py_RETURN_NONE;
}

static PyObject* View_SetImage(ViewObj *self, PyObject *args)
{
  const char *name;
  PyObject *image;

  if (!PyArg_ParseTuple(args, "sO:set_image", &name, &image))
    return NULL;

  if (PyArray_Check(image)) {
    image = Image_FromNumpyArray(image);
    PyObject *args2 = Py_BuildValue("sO", name, image);
    PyObject *result = View_SetImage(self, args2);
    Py_DECREF(args2);
    Py_DECREF(image);
    return result;
  }

  mve::ImageBase::Ptr ptr = Image_GetImageBasePtr(image);
  if (ptr == NULL) {
    return PyErr_Format(PyExc_TypeError,
                        "2nd argument should be %s or numpy.ndarray",
                        Image_Type()->tp_name);
  }

  self->thisptr->set_image(name, ptr);

  Py_RETURN_NONE;
}

static PyObject* View_RemoveImage(ViewObj *self, PyObject *arg)
{
  if (PyUnicode_Check(arg)) {
    PyObject *bytes = PyUnicode_AsUTF8String(arg);
    PyObject *result = View_RemoveImage(self, bytes);
    Py_DECREF(bytes);
    return result;
  }

  const char* name = PyBytes_AsString(arg);
  if (!name)
    return NULL;

  self->thisptr->remove_embedding(name);

  Py_RETURN_NONE;
}

static PyObject* View_Load(ViewObj *self, PyObject *args, PyObject *kwds)
{
  char* klist[] = { "filename", "merge", NULL };
  const char* filename = NULL;
  PyObject *merge = Py_False;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|O!:load", klist, &filename, &PyBool_Type, &merge))
    return NULL;

  try {
    bool bmerge = PyLong_AsLong(merge);
    self->thisptr->load_mve_file(filename, bmerge); // other args: merge
  } catch (const util::Exception& e) {
    PyErr_SetString(PyExc_Exception, e.what());
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject* View_Save(ViewObj *self, PyObject *args, PyObject *kwds)
{
  char* klist[] = { "filename", NULL };
  const char* filename = NULL;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|s:save", klist, &filename))
    return NULL;

  try {
    if (filename) {
      self->thisptr->save_mve_file_as(filename);
    } else {
      self->thisptr->save_mve_file(); // other args: rebuild
    }
  } catch (const util::Exception& e) {
    PyErr_SetString(PyExc_Exception, e.what());
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject* View_Reload(ViewObj *self, PyObject *args, PyObject *kwds)
{
  char* klist[] = { "merge", NULL };
  PyObject *merge = Py_False;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O!:reload", klist, &PyBool_Type, &merge))
    return NULL;

  try {
    bool bmerge = PyLong_AsLong(merge);
    self->thisptr->reload_mve_file(bmerge);
  } catch (const util::Exception& e) {
    PyErr_SetString(PyExc_Exception, e.what());
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyMethodDef View_methods[] = {
  {"cleanup_cache", (PyCFunction)View_CleanupCache, METH_NOARGS, "Clean Cache"},
  {"has_image", (PyCFunction)View_HasImage, METH_O, "Check if image embedding exists"},
  {"get_image", (PyCFunction)View_GetImage, METH_O, "Get an image embedding"},
  {"set_image", (PyCFunction)View_SetImage, METH_VARARGS, "Set an image embedding"},
  {"remove_image", (PyCFunction)View_RemoveImage, METH_O, "Remove an image embedding"},
  {"load", (PyCFunction)View_Load, METH_VARARGS|METH_KEYWORDS, "Load"},
  {"save", (PyCFunction)View_Save, METH_VARARGS|METH_KEYWORDS, "Save"},
  {"reload", (PyCFunction)View_Reload, METH_VARARGS|METH_KEYWORDS, "Reload"},
  {NULL, NULL, 0, NULL}
};

static PyObject* View_GetId(ViewObj *self, void* closure)
{
  return PyLong_FromSize_t(self->thisptr->get_id());
}

static int View_SetId(ViewObj *self, PyObject *value, void* closure)
{
  self->thisptr->set_id(PyLong_AsSsize_t(value));
  return 0;
}

static PyObject* View_GetName(ViewObj *self, void* closure)
{
  return PyString_FromString(self->thisptr->get_name().c_str());
}

static int View_SetName(ViewObj *self, PyObject *value, void* closure)
{
  if (PyUnicode_Check(value)) {
    PyObject *bytes = PyUnicode_AsUTF8String(value);
    int result = View_SetName(self, bytes, closure);
    Py_DECREF(bytes);
    return result;
  }

  const char * name = PyBytes_AsString(value);
  if (!name)
    return -1;

  self->thisptr->set_name(name);

  return 0;
}

static PyObject* View_GetCamera(ViewObj *self, void* closure)
{
  return CameraInfoObj_Create(self->thisptr->get_camera());
}

static PyObject* View_IsValid(ViewObj *self, void* closure)
{
  if (self->thisptr->is_camera_valid()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject* View_GetFilename(ViewObj *self, void* closure)
{
  return PyString_FromString(self->thisptr->get_filename().c_str());
}

static PyGetSetDef View_getset[] = {
  {"id", (getter)View_GetId, (setter)View_SetId, "ID", NULL },
  {"name", (getter)View_GetName, (setter)View_SetName, "Name", NULL},
  {"camera", (getter)View_GetCamera, NULL, "Camera", NULL},
  {"valid", (getter)View_IsValid, NULL, "Is Camera Valid", NULL},
  {"filename", (getter)View_GetFilename, NULL, "Filename", NULL},
  {NULL, NULL, NULL, NULL, NULL}
};

static int View_Init(ViewObj *self, PyObject *args, PyObject *kwds)
{
  char* klist[] = { "filename", NULL };
  const char* filename = NULL;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|s:init", klist, &filename))
    return -1;

  self->thisptr = mve::View::create();

  if (filename) {
    try {
      self->thisptr->load_mve_file(filename);
    } catch (const util::Exception& e) {
      PyErr_SetString(PyExc_Exception, e.what());
      return -1;
    }
  }

  return 0;
}

static PyObject* View_New(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
  ViewObj* self = (ViewObj*) subtype->tp_alloc(subtype, 0);

  if (self != NULL) {
    ::new(&(self->thisptr)) mve::View::Ptr();
  }

  return (PyObject*) self;
}

static void View_Dealloc(ViewObj *self)
{
  //printf("view %d is deallocated\n", self->thisptr->get_id());
  self->thisptr.reset();
  Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyObject* View_Representation(ViewObj *self)
{
  return PyString_FromFormat("View(id=%zu, name=%s)",
                             self->thisptr->get_id(),
                             self->thisptr->get_name().c_str());
}

static PyTypeObject ViewType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "mve.core.View", // tp_name
  sizeof(ViewObj), // tp_basicsize
  0, // tp_itemsize
  (destructor)View_Dealloc, // tp_dealloc
  0, // tp_print
  0, // tp_getattr (deprecated)
  0, // tp_setattr (deprecated)
#if PY_MAJOR_VERSION < 3
  0, // tp_compare
#else
  0, // reserved
#endif
  (reprfunc)View_Representation, // tp_repr
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
  "MVE View", // tp_doc
  0, // tp_traverse
  0, // tp_clear
  0, // tp_richcompare
  0, // tp_weaklistoffset
  0, // tp_iter
  0, // tp_iternext
  View_methods, // tp_methods
  0, // tp_members
  View_getset, // tp_getset
  0, // tp_base
  0, // tp_dict
  0, // tp_descr_get
  0, // tp_descr_set
  0, // tp_dictoffset
  (initproc)View_Init, // tp_init
  0, // tp_alloc
  (newfunc)View_New, // tp_new
  0, // tp_free
  0, // tp_is_gc
};

/***************************************************************************
 *
 *
 */

PyObject* ViewObj_Create(mve::View::Ptr ptr)
{
  PyObject* args = PyTuple_New(0);
  PyObject* kwds = PyDict_New();
  PyObject* obj = ViewType.tp_new(&ViewType, args, kwds);
  Py_DECREF(args);
  Py_DECREF(kwds);

  ((ViewObj*) obj)->thisptr = ptr;

  return obj;
}

void load_View(PyObject* mod)
{
  if (PyType_Ready(&ViewType) < 0)
    abort();
  Py_INCREF(&ViewType);

  PyModule_AddObject(mod, "View", (PyObject*)&ViewType);
}
