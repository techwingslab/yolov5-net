using System.Collections.Generic;

namespace Yolov5Net.Scorer.Models.Abstract;

/// <summary>
/// Model descriptor.
/// </summary>
public record YoloModel(
    int Width,
    int Height,
    int Depth,

    int Dimensions,

    int[] Strides,
    int[][][] Anchors,
    int[] Shapes,

    float Confidence,
    float MulConfidence,
    float Overlap,

    string[] Outputs,
    List<YoloLabel> Labels,
    bool UseDetect
);
