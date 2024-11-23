// Machine Learning
// File name: Program.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;
using System.Threading.Channels;

using MachineLearning;
using MachineLearning.NeuralNetwork.LearningRates;
using MachineLearning.Typed.NeuralNetwork.Layers;
using MachineLearning.Typed.NeuralNetwork.Losses;
using MachineLearning.Typed.NeuralNetwork.Operations;
using MachineLearning.Typed.NeuralNetwork.Optimizers;

using static MachineLearning.Typed.ArrayUtils;

// Prepare data:
// 3 (none, vertical, and horizontal stripes) * 4 examples * 1 channel * 3 rows * 3 columns

const int inputChannels = 1;
const int examples = 12;

float[,,,] xTrain = new float[examples, inputChannels, 3, 3]
{
    // neither vertical nor horizonta
    {
        {
            { 0, 0, 0 },
            { 0, 0, 0 },
            { 0, 0, 0 }
        }
    },
    {
        {
            { 1, 1, 1 },
            { 1, 1, 1 },
            { 1, 1, 1 }
        }
    },
    {
        {
            { 1, 0, 1 },
            { 0, 1, 0 },
            { 1, 0, 1 }
        }
    },
    {
        {
            { 0, 1, 0 },
            { 1, 1, 1 },
            { 0, 1, 0 },
        }
    },
    // horizontal
    {
        {
            { 1, 1, 1 },
            { 0, 0, 0 },
            { 1, 1, 1 }
        }
    },
    {
        {
            { 1, 1, 1 },
            { 0, 0, 0 },
            { 0, 0, 0 }
        }
    },
    {
        {
            { 0, 0, 0 },
            { 0, 0, 0 },
            { 1, 1, 1 }
        }
    },
    {
        {
            { 0, 0, 0 },
            { 1, 1, 1 },
            { 0, 0, 0 },
        }
    },
    // vertical
    {
        {
            { 1, 0, 1 },
            { 1, 0, 1 },
            { 1, 0, 1 }
        }
    },
    {
        {
            { 0, 1, 0 },
            { 0, 1, 0 },
            { 0, 1, 0 }
        }
    },
    {
        {
            { 1, 0, 0 },
            { 1, 0, 0 },
            { 1, 0, 0 }
        }
    },
    {
        {
            { 0, 0, 1 },
            { 0, 0, 1 },
            { 0, 0, 1 },
        }
    }
};

// Values - 0 for unknown, 1 for horizontal stripes, 2 for vertical stripes
float[,] oneHot = new float[examples, 3] {
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, 1 },
};

float loss = Train(xTrain, oneHot, 100);

Console.WriteLine($"loss: {loss}");
Console.ReadLine();

static float Train(float[,,,] xTrain, float[,] yTrain, int iterations = 2_000 /*, float learningRate = 0.01f */, int? seed = null)
{
    float loss = 0;
    Random random;
    if (seed.HasValue)
        random = new(seed.Value);
    else
        random = new();

    const int outputChannels = 2;
    float[,,,] kernels = CreateRandom(inputChannels, outputChannels, 3, 3, random);
    Conv2D conv2D = new(kernels);
    Tanh4D tanh4D = new();
    Flatten flatten = new();

    float[,] weights = CreateRandom(outputChannels * 3 * 3, 10, random);
    WeightMultiply weightMultiply = new(weights);

    float[] bias = CreateRandom(10, random);
    BiasAdd biasAdd = new(bias);

    SoftmaxCrossEntropyLoss softmaxCrossEntropyLoss = new();

    LearningRate learningRate = new ExponentialDecayLearningRate(0.19f, 0.05f);
    Optimizer optimizer = new StochasticGradientDescent(learningRate);

    for (int i = 0; i < iterations; i++)
    {
        (xTrain, yTrain) = PermuteData(xTrain, yTrain, random);

        // Forward

        float[,,,] conv2DOutput = conv2D.Forward(xTrain, false);
        float[,,,] tanh4DOutput = tanh4D.Forward(conv2DOutput, false);
        float[,] flattenOutput = flatten.Forward(tanh4DOutput, false);
        float[,] weightMultiplyOutput = weightMultiply.Forward(flattenOutput, false);
        float[,] biasAddOutput = biasAdd.Forward(weightMultiplyOutput, false);

        loss = softmaxCrossEntropyLoss.Forward(biasAddOutput, yTrain);

        // Print loss every 10 steps

        if (i % 10 == 0)
        {
            Console.WriteLine($"iteration: {i}, loss: {loss}");
        }

        // Backward

        float[,] softmaxCrossEntropyLossGradient = softmaxCrossEntropyLoss.Backward();

        float[,] biasAddGradient = biasAdd.Backward(softmaxCrossEntropyLossGradient);
        float[,] weightMultiplyGradient = weightMultiply.Backward(biasAddGradient);
        float[,,,] flattenGradient = flatten.Backward(weightMultiplyGradient);
        float[,,,] tanh4DGradient = tanh4D.Backward(flattenGradient);
        float[,,,] conv2DGradient = conv2D.Backward(tanh4DGradient);

        // Update weights

        optimizer.UpdateLearningRate(i + 1, iterations);
        conv2D.UpdateParams(null, optimizer);
        weightMultiply.UpdateParams(null, optimizer);
        biasAdd.UpdateParams(null, optimizer);
    }

    return loss;
}

