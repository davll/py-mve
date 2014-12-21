#include "scene.h"
#include <mve/scene.h>
#include <Python.h>
#include <structmember.h>
#include "view.h"

#if PY_MAJOR_VERSION >= 3
#  define PyString_FromString PyUnicode_FromString
#  define PyString_FromFormat PyUnicode_FromFormat
#endif

#pragma GCC diagnostic ignored "-Wc++11-compat-deprecated-writable-strings"

/***************************************************************************
 * Scene Object
 *
 */

struct SceneObj {
  PyObject_HEAD
  mve::Scene::Ptr thisptr;
  //PyObject *viewlist;
};

static PyObject* Scene_Load(SceneObj *self, PyObject *arg)
{
  if (PyUnicode_Check(arg)) {
    PyObject *bytes = PyUnicode_AsUTF8String(arg);
    PyObject *result = Scene_Load(self, bytes);
    Py_DECREF(bytes);
    return result;
  }

  const char* path = PyBytes_AsString(arg);
  if (!path)
    return NULL;

  self->thisptr->load_scene(path);

  Py_RETURN_NONE;
}

static PyObject* Scene_CleanupCache(SceneObj *self)
{
  self->thisptr->cache_cleanup();
  Py_RETURN_NONE;
}

static PyObject* Scene_GetViews(SceneObj *self, void* closure)
{
  mve::Scene::ViewList& views = self->thisptr->get_views();

  size_t n = views.size();
  PyObject* list = PyList_New(n);
  for (size_t i = 0; i < n; ++i) {
    PyList_SetItem(list, i, ViewObj_Create(views[i]));
  }

  return list;
}

static PyObject* Scene_GetPath(SceneObj *self, void* closure)
{
  return PyString_FromString(self->thisptr->get_path().c_str());
}

static PyMethodDef Scene_methods[] = {
  {"load", (PyCFunction)Scene_Load, METH_O, "Load Scene"},
  {"cleanup_cache", (PyCFunction)Scene_CleanupCache, METH_NOARGS, "Clean Cache"},
  {NULL, NULL, 0, NULL}
};

static PyGetSetDef Scene_getset[] = {
  {"views", (getter)Scene_GetViews, NULL, "Views", NULL },
  {"path", (getter)Scene_GetPath, NULL, "Base Path", NULL},
  {NULL, NULL, NULL, NULL, NULL}
};

static int Scene_Init(SceneObj *self, PyObject *args, PyObject *kwds)
{
  char* klist[] = { "path", NULL };
  const char* path = NULL;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|s:init", klist, &path))
    return -1;

  if (path) {
    self->thisptr = mve::Scene::create(path);
  } else {
    self->thisptr = mve::Scene::create();
  }

  return 0;
}

static PyObject* Scene_New(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
  SceneObj* self = (SceneObj*) subtype->tp_alloc(subtype, 0);

  if (self != NULL) {
    ::new(&(self->thisptr)) mve::Scene::Ptr();
  }

  return (PyObject*) self;
}

static void Scene_Dealloc(SceneObj *self)
{
  self->thisptr.reset();
  Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyObject* Scene_Representation(SceneObj *self)
{
  return PyString_FromFormat("Scene(path=%s)",
                             self->thisptr->get_path().c_str());
}

static PyTypeObject SceneType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "mve.core.Scene", // tp_name
  sizeof(SceneObj), // tp_basicsize
  0, // tp_itemsize
  (destructor)Scene_Dealloc, // tp_dealloc
  0, // tp_print
  0, // tp_getattr
  0, // tp_setattr
#if PY_MAJOR_VERSION < 3
  0, // tp_compare
#else
  0, // reserved
#endif
  (reprfunc)Scene_Representation, // tp_repr
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
  "MVE Scene", // tp_doc
  0, // tp_traverse
  0, // tp_clear
  0, // tp_richcompare
  0, // tp_weaklistoffset
  0, // tp_iter
  0, // tp_iternext
  Scene_methods, // tp_methods
  0, // tp_members
  Scene_getset, // tp_getset
  0, // tp_base
  0, // tp_dict
  0, // tp_descr_get
  0, // tp_descr_set
  0, // tp_dictoffset
  (initproc)Scene_Init, // tp_init
  0, // tp_alloc
  (newfunc)Scene_New, // tp_new
  0, // tp_free
  0, // tp_is_gc
};

void load_Scene(PyObject* mod)
{
  if (PyType_Ready(&SceneType) < 0)
    abort();
  Py_INCREF(&SceneType);

  PyModule_AddObject(mod, "Scene", (PyObject*)&SceneType);
}
