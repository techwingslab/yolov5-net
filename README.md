# Yolov5Net
YOLOv5 object detection with ML.NET, ONNX

![example](https://github.com/mentalstack/yolov5-net/blob/master/img/result.jpg?raw=true)

## Installation

Run this line from Package Manager Console:

```
Install-Package Yolov5Net -Version 1.0.9
```

For CPU usage run this line from Package Manager Console:

```
Install-Package Microsoft.ML.OnnxRuntime -Version 1.9.0
```

For GPU usage run this line from Package Manager Console:

```
Install-Package Microsoft.ML.OnnxRuntime.Gpu -Version 1.9.0
```

CPU and GPU packages can't be installed together.

## Usage

Yolov5Net contains two COCO pre-defined models: YoloCocoP5Model, YoloCocoP6Model. 

If you have custom trained model, then inherit from YoloModel and override all the required properties and methods. See YoloCocoP5Model or YoloCocoP6Model implementation to get know how to wrap your own model. 

```c#
using var image = Image.FromFile("Assets/test.jpg");

using var scorer = new YoloScorer<YoloCocoP5Model>("Assets/Weights/yolov5s.onnx");

List<YoloPrediction> predictions = scorer.Predict(image);

using var graphics = Graphics.FromImage(image);

foreach (var prediction in predictions) // iterate predictions to draw results
{
	double score = Math.Round(prediction.Score, 2);

	graphics.DrawRectangles(new Pen(prediction.Label.Color, 1),
		new[] { prediction.Rectangle });

	var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);

	graphics.DrawString($"{prediction.Label.Name} ({score})",
		new Font("Arial", 16, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),
		new PointF(x, y));
}

image.Save("Assets/result.jpg");
```

