import os
import sys
import shutil
import uuid
from os.path import join, dirname, realpath, exists
import tensorflow as tf

if sys.platform =="linux":
  tf.app.flags.DEFINE_string('oplib_name', 'aster', 'Name of op library.')
  tf.app.flags.DEFINE_string('oplib_suffix', '.so', 'Library suffix.')
else:
  tf.app.flags.DEFINE_string('oplib_name', 'reverse_string', 'Name of op library.')
  tf.app.flags.DEFINE_string('oplib_suffix', '.dll', 'Library suffix.')

FLAGS = tf.app.flags.FLAGS


def _load_oplib_windows(lib_name):
  lib_path = join(dirname(realpath(__file__)), '{0}{1}'.format(lib_name, FLAGS.oplib_suffix))
  assert exists(lib_path), '{0} not found'.format(lib_path)
  # duplicate library with a random new name so that
  # a running program will not be interrupted when the lib file is updated
  # lib_copy_path = '/tmp/lib{0}_{1}{2}'.format(lib_name, str(uuid.uuid4())[:8], FLAGS.oplib_suffix)
  lib_copy_path = os.path.join("C:",os.path.sep,"Windows","Temp",'lib{0}_{1}{2}'.format(lib_name, str(uuid.uuid4())[:8], FLAGS.oplib_suffix))
  shutil.copyfile(lib_path, lib_copy_path)
  oplib = tf.load_op_library(lib_copy_path)
  return oplib




def _load_oplib_linux(lib_name):
  """
  Load TensorFlow operator library.
  """
  lib_path = join(dirname(realpath(__file__)), 'lib{0}{1}'.format(lib_name, FLAGS.oplib_suffix))
  assert exists(lib_path), '{0} not found'.format(lib_path)

  # duplicate library with a random new name so that
  # a running program will not be interrupted when the lib file is updated
  lib_copy_path = '/tmp/lib{0}_{1}{2}'.format(lib_name, str(uuid.uuid4())[:8], FLAGS.oplib_suffix)
  # lib_copy_path = 'lib{0}_{1}{2}'.format(lib_name, str(uuid.uuid4())[:8], FLAGS.oplib_suffix)
  shutil.copyfile(lib_path, lib_copy_path)
  oplib = tf.load_op_library(lib_copy_path)
  return oplib

_oplib = None

if sys.platform == "linux":
  _oplib = _load_oplib_linux(FLAGS.oplib_name)
else:
  _oplib = _load_oplib_windows(FLAGS.oplib_name)

if _oplib is not None:
  string_reverse = _oplib.string_reverse
else:
  print("Error reading library check ops file")
# map C++ operators to python objects
# string_filtering = _oplib.string_filtering
# string_reverse = _oplib.string_reverse
# divide_curve = _oplib.divide_curve
