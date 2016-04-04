package main

// #cgo CXXFLAGS: -std=c++11
// #cgo darwin pkg-config: opencv
// #cgo linux  pkg-config: opencv
import "C"

import (
	"fmt"
	"path"
	"runtime"

	"github.com/ctava/go-opencv-face-detection/opencv"
	"github.com/lazywei/go-opencv/opencv"
)

func main() {
	fmt.Println("hello world")

	_, currentfile, _, _ := runtime.Caller(0)
	image := opencv.LoadImage(path.Join(path.Dir(currentfile), "./99.jpg"))

	facedetector := opencv.NewFaceDetector()
	facedetector.InitCascadeClassifiers()

	face := facedetector.GetPreprocessedFace(image)
	opencv.SaveImage("./face.jpg", face, 0)
}
