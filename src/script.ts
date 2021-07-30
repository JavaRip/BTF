import { Matrix } from 'ml-matrix';

class NeuralNetwork {
  inputNodes: number;
  hiddenNodes: number;
  outputNodes: number;
  learningRate: number;
  inputToHiddenWeights: Matrix;
  hiddenToOutputWeights: Matrix;
  boundMultiplyByRand: any;

  constructor(inputNodes: number, hiddenNodes: number, outputNodes: number, learningRate: number) {
    // bound functions
    this.boundMultiplyByRand = this.multiplyByRand.bind(this);

    // initial attributes
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;
    this.learningRate = learningRate;

    // calculate random starting weight matrices
    this.inputToHiddenWeights = this.matrixFunc(Matrix.ones(this.hiddenNodes, this.inputNodes), this.boundMultiplyByRand, [-0.5, 1]);
    this.hiddenToOutputWeights = this.matrixFunc(Matrix.ones(this.outputNodes, this.hiddenNodes), this.boundMultiplyByRand, [-0.5, 1]);
  }

  train(input: Matrix): string {
    // const hiddenInputs: Matrix = this.inputToHiddenWeights.mmul(input);
    // const hiddenOutputs: Matrix =
    console.log(input);
    return '🏋️';
  }

  query(): string {
    return '🧐';
  }

  matrixFunc(matrix: Matrix, f: any, fParams: number[] = []): Matrix {
    const newMatrix: number[][] = [];
    for (let row = 0; row < matrix.rows; row += 1) {
      const column: number[] = [];
      for (let col = 0; col < matrix.columns; col += 1) {
        column.push(f(matrix.get(row, col), ...fParams));
      }
      newMatrix.push(column);
    }
    return new Matrix(newMatrix);
  }

  multiplyByRand(x: number, offset: number, maxValue: number): number {
    return x * Math.random() * maxValue + offset;
  }

  sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }
}

const neuralNetwork = new NeuralNetwork(3, 3, 3, 3);
neuralNetwork.matrixFunc(neuralNetwork.inputToHiddenWeights, console.log);
console.log(neuralNetwork);
