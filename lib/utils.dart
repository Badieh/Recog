import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';

import 'package:google_ml_kit/google_ml_kit.dart';
//import 'package:google_ml_vision/google_ml_vision.dart' as vision;
import 'package:image/image.dart' as img;

typedef HandleDetection = Future<dynamic> Function(InputImage image);
enum Choice { view, delete }

Future<CameraDescription> getCamera(CameraLensDirection dir) async {
  return await availableCameras().then(
    (List<CameraDescription> cameras) => cameras.firstWhere(
      (CameraDescription camera) => camera.lensDirection == dir,
    ),
  );
}

// InputImageData buildMetaData(
//   CameraImage image,
//   InputImageRotation rotation,
// ) {
//   return InputImageData(
//     inputImageFormat: InputImageFormat.BGRA8888,
//     size: Size(image.width.toDouble(), image.height.toDouble()),
//     imageRotation: rotation,
//     planeData: image.planes.map(
//       (Plane plane) {
//         return InputImagePlaneMetadata(
//           bytesPerRow: plane.bytesPerRow,
//           height: plane.height,
//           width: plane.width,
//         );
//       },
//     ).toList(),
//   );
// }

// Future<dynamic> detect(
//   CameraImage image,
//   HandleDetection handleDetection,
//   InputImageRotation rotation,
// ) async {
//   return handleDetection(
//     InputImage.fromBytes(
//       bytes: image.planes[0].bytes,
//       metadata: buildMetaData(image, rotation),
//     ),
//   );
// }
Future<List<Face>> detect(CameraImage image, InputImageRotation rotation) {

final faceDetector = GoogleMlKit.vision.faceDetector(
   FaceDetectorOptions(
    performanceMode: FaceDetectorMode.accurate,
    enableLandmarks: true,
  ), 
);
final WriteBuffer allBytes = WriteBuffer();
for (final Plane plane in image.planes) {
  allBytes.putUint8List(plane.bytes);
}
final bytes = allBytes.done().buffer.asUint8List();

final Size imageSize =
    Size(image.width.toDouble(), image.height.toDouble());
final inputImageFormat =
    InputImageFormatValue.fromRawValue(image.format.raw) ??
        InputImageFormat.nv21;

final inputImageData = InputImageMetadata(
  size: imageSize,
  rotation: rotation,
  format: inputImageFormat,
  bytesPerRow: image.planes[0].bytesPerRow,
);



return  faceDetector.processImage(
  InputImage.fromBytes(
    bytes: bytes,
  metadata: inputImageData,
  ),
);
}


InputImageRotation rotationIntToImageRotation(int rotation) {
  switch (rotation) {
    case 0:
      return InputImageRotation.rotation0deg ;
    case 90:
      return InputImageRotation.rotation90deg;
    case 180:
      return InputImageRotation.rotation180deg;
    default:
      assert(rotation == 270);
      return InputImageRotation.rotation270deg;
  }
}

Float32List imageToByteListFloat32(
    img.Image image, int inputSize, double mean, double std) {
  var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
  var buffer = Float32List.view(convertedBytes.buffer);
  int pixelIndex = 0;
  for (var i = 0; i < inputSize; i++) {
    for (var j = 0; j < inputSize; j++) {
      var pixel = image.getPixel(j, i);
      
      buffer[pixelIndex++] = (pixel.r - mean) / std;
      buffer[pixelIndex++] = (pixel.g - mean) / std;
      buffer[pixelIndex++] = (pixel.b - mean) / std;
    }
  }
  return convertedBytes.buffer.asFloat32List();
}

double euclideanDistance(List e1, List e2) {
  double sum = 0.0;
  for (int i = 0; i < e1.length; i++) {
    sum += pow((e1[i] - e2[i]), 2);
  }
  return sqrt(sum);
}
