/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.8
 *
 * This file is not intended to be easily readable and contains a number of
 * coding conventions designed to improve portability and efficiency. Do not make
 * changes to this file unless you know what you are doing--modify the SWIG
 * interface file instead.
 * ----------------------------------------------------------------------------- */

// source: facepreprocessor.swig

package opencv

import "unsafe"

import "sync"

var _cgo_runtime_cgocall func(unsafe.Pointer, uintptr)

type _ unsafe.Pointer

var Swig_escape_always_false bool
var Swig_escape_val interface{}

type _swig_fnptr *byte
type _swig_memberptr *byte

type _ sync.Mutex

var _wrap_Swig_free_opencv_72ff6b21c13f0bb9 unsafe.Pointer

func _swig_wrap_Swig_free(base uintptr) {
	_swig_p := uintptr(unsafe.Pointer(&base))
	_cgo_runtime_cgocall(_wrap_Swig_free_opencv_72ff6b21c13f0bb9, _swig_p)
	return
}

func Swig_free(arg1 uintptr) {
	_swig_wrap_Swig_free(arg1)
}

type SwigcptrFacePreprocessor uintptr

func (p SwigcptrFacePreprocessor) Swigcptr() uintptr {
	return (uintptr)(p)
}

func (p SwigcptrFacePreprocessor) SwigIsFacePreprocessor() {
}

var _wrap_FacePreprocessor_initCascadeClassifiers_facepreprocessor_72ff6b21c13f0bb9 unsafe.Pointer

func _swig_wrap_FacePreprocessor_initCascadeClassifiers(base SwigcptrFacePreprocessor) {
	_swig_p := uintptr(unsafe.Pointer(&base))
	_cgo_runtime_cgocall(_wrap_FacePreprocessor_initCascadeClassifiers_facepreprocessor_72ff6b21c13f0bb9, _swig_p)
	return
}

func (arg1 SwigcptrFacePreprocessor) InitCascadeClassifiers() {
	_swig_wrap_FacePreprocessor_initCascadeClassifiers(arg1)
}

var _wrap_FacePreprocessor_getPreprocessedFace_facepreprocessor_72ff6b21c13f0bb9 unsafe.Pointer

func _swig_wrap_FacePreprocessor_getPreprocessedFace(base SwigcptrFacePreprocessor, _ uintptr) (_ SwigcptrCv_Mat) {
	_swig_p := uintptr(unsafe.Pointer(&base))
	_cgo_runtime_cgocall(_wrap_FacePreprocessor_getPreprocessedFace_facepreprocessor_72ff6b21c13f0bb9, _swig_p)
	return
}

func (arg1 SwigcptrFacePreprocessor) GetPreprocessedFace(arg2 Cv_Mat) (_swig_ret Cv_Mat) {
	return _swig_wrap_FacePreprocessor_getPreprocessedFace(arg1, arg2.Swigcptr())
}

var _wrap_new_FacePreprocessor_facepreprocessor_72ff6b21c13f0bb9 unsafe.Pointer

func _swig_wrap_new_FacePreprocessor() (base SwigcptrFacePreprocessor) {
	_swig_p := uintptr(unsafe.Pointer(&base))
	_cgo_runtime_cgocall(_wrap_new_FacePreprocessor_facepreprocessor_72ff6b21c13f0bb9, _swig_p)
	return
}

func NewFacePreprocessor() (_swig_ret FacePreprocessor) {
	return _swig_wrap_new_FacePreprocessor()
}

var _wrap_delete_FacePreprocessor_facepreprocessor_72ff6b21c13f0bb9 unsafe.Pointer

func _swig_wrap_delete_FacePreprocessor(base uintptr) {
	_swig_p := uintptr(unsafe.Pointer(&base))
	_cgo_runtime_cgocall(_wrap_delete_FacePreprocessor_facepreprocessor_72ff6b21c13f0bb9, _swig_p)
	return
}

func DeleteFacePreprocessor(arg1 FacePreprocessor) {
	_swig_wrap_delete_FacePreprocessor(arg1.Swigcptr())
}

type FacePreprocessor interface {
	Swigcptr() uintptr
	SwigIsFacePreprocessor()
	InitCascadeClassifiers()
	GetPreprocessedFace(arg2 Cv_Mat) (_swig_ret Cv_Mat)
}

type SwigcptrCv_Mat uintptr
type Cv_Mat interface {
	Swigcptr() uintptr
}

func (p SwigcptrCv_Mat) Swigcptr() uintptr {
	return uintptr(p)
}

type SwigcptrVoid uintptr
type Void interface {
	Swigcptr() uintptr
}

func (p SwigcptrVoid) Swigcptr() uintptr {
	return uintptr(p)
}
