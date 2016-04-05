%module opencv
%{
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/core/version.hpp>
#include "facedetector.h"
%}

%include <typemaps.i>
%include "std_string.i"
%include "std_vector.i"

%include "facedetector.h"