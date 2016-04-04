%module opencv
%{
#include "opencv2/core/core.hpp"
#include "facedetector.h"
%}

%include <typemaps.i>
%include "std_string.i"
%include "std_vector.i"

%include "facedetector.h"