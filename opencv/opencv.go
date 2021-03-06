/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.8
 *
 * This file is not intended to be easily readable and contains a number of
 * coding conventions designed to improve portability and efficiency. Do not make
 * changes to this file unless you know what you are doing--modify the SWIG
 * interface file instead.
 * ----------------------------------------------------------------------------- */

// source: facedetector.swig

package opencv

import "unsafe"
import _ "runtime/cgo"
import "sync"

var _cgo_runtime_cgocall func(unsafe.Pointer, uintptr)



type _ unsafe.Pointer



var Swig_escape_always_false bool
var Swig_escape_val interface{}


type _swig_fnptr *byte
type _swig_memberptr *byte


type _ sync.Mutex

var _wrap_Swig_free_opencv_c836319826f53285 unsafe.Pointer

func _swig_wrap_Swig_free(base uintptr) {
	_swig_p := uintptr(unsafe.Pointer(&base))
	_cgo_runtime_cgocall(_wrap_Swig_free_opencv_c836319826f53285, _swig_p)
	return
}

func Swig_free(arg1 uintptr) {
	_swig_wrap_Swig_free(arg1)
}

type SwigcptrFaceDetector uintptr

func (p SwigcptrFaceDetector) Swigcptr() uintptr {
	return (uintptr)(p)
}

func (p SwigcptrFaceDetector) SwigIsFaceDetector() {
}

var _wrap_FaceDetector_initCascadeClassifiers_opencv_c836319826f53285 unsafe.Pointer

func _swig_wrap_FaceDetector_initCascadeClassifiers(base SwigcptrFaceDetector) {
	_swig_p := uintptr(unsafe.Pointer(&base))
	_cgo_runtime_cgocall(_wrap_FaceDetector_initCascadeClassifiers_opencv_c836319826f53285, _swig_p)
	return
}

func (arg1 SwigcptrFaceDetector) InitCascadeClassifiers() {
	_swig_wrap_FaceDetector_initCascadeClassifiers(arg1)
}

var _wrap_FaceDetector_getPreprocessedFace_opencv_c836319826f53285 unsafe.Pointer

func _swig_wrap_FaceDetector_getPreprocessedFace(base SwigcptrFaceDetector, _ uintptr) (_ SwigcptrCv_Mat) {
	_swig_p := uintptr(unsafe.Pointer(&base))
	_cgo_runtime_cgocall(_wrap_FaceDetector_getPreprocessedFace_opencv_c836319826f53285, _swig_p)
	return
}

func (arg1 SwigcptrFaceDetector) GetPreprocessedFace(arg2 Cv_Mat) (_swig_ret Cv_Mat) {
	return _swig_wrap_FaceDetector_getPreprocessedFace(arg1, arg2.Swigcptr())
}

var _wrap_new_FaceDetector_opencv_c836319826f53285 unsafe.Pointer

func _swig_wrap_new_FaceDetector() (base SwigcptrFaceDetector) {
	_swig_p := uintptr(unsafe.Pointer(&base))
	_cgo_runtime_cgocall(_wrap_new_FaceDetector_opencv_c836319826f53285, _swig_p)
	return
}

func NewFaceDetector() (_swig_ret FaceDetector) {
	return _swig_wrap_new_FaceDetector()
}

var _wrap_delete_FaceDetector_opencv_c836319826f53285 unsafe.Pointer

func _swig_wrap_delete_FaceDetector(base uintptr) {
	_swig_p := uintptr(unsafe.Pointer(&base))
	_cgo_runtime_cgocall(_wrap_delete_FaceDetector_opencv_c836319826f53285, _swig_p)
	return
}

func DeleteFaceDetector(arg1 FaceDetector) {
	_swig_wrap_delete_FaceDetector(arg1.Swigcptr())
}

type FaceDetector interface {
	Swigcptr() uintptr
	SwigIsFaceDetector()
	InitCascadeClassifiers()
	GetPreprocessedFace(arg2 Cv_Mat) (_swig_ret Cv_Mat)
}


type SwigcptrCv_Mat uintptr
type Cv_Mat interface {
	Swigcptr() uintptr;
}
func (p SwigcptrCv_Mat) Swigcptr() uintptr {
	return uintptr(p)
}

type SwigcptrVoid uintptr
type Void interface {
	Swigcptr() uintptr;
}
func (p SwigcptrVoid) Swigcptr() uintptr {
	return uintptr(p)
}

