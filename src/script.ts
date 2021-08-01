import { Matrix } from 'ml-matrix';

class NeuralNetwork {
  inNodes: number;
  hidNodes: number;
  outNodes: number;
  learningRate: number;
  inToHidWeights: Matrix;
  hidToOutWeights: Matrix;
  boundTimesByRand: (x: number, offset: number, maxValue: number) => number;
  boundSigmoid: (x: number) => number;

  constructor(inputNodes: number, hiddenNodes: number, outputNodes: number, learningRate: number) {
    // bound functions
    this.boundTimesByRand = this.timesByRand.bind(this);
    this.boundSigmoid = this.sigmoid.bind(this);

    // initial attributes
    this.inNodes = inputNodes;
    this.hidNodes = hiddenNodes;
    this.outNodes = outputNodes;
    this.learningRate = learningRate;

    // calculate random starting weight matrices
    const baseMatrix = Matrix.ones(this.hidNodes, this.inNodes);
    this.inToHidWeights = this.matrixFunc(baseMatrix, this.boundTimesByRand, [-0.5, 1]);
    this.hidToOutWeights = this.matrixFunc(baseMatrix, this.boundTimesByRand, [-0.5, 1]);
  }

  train(): string {
    return '🏋️';
  }

  query(input: Matrix): Matrix {
    // calculate signals into hidden layer, then output of hidden layer
    const hidIn: Matrix = this.inToHidWeights.mmul(input);
    const hidOut: Matrix = this.matrixFunc(hidIn, this.boundSigmoid);

    // calculate signals into output layer, then output of output layer
    const outIn: Matrix = this.hidToOutWeights.mmul(hidOut);
    const outOut: Matrix = this.matrixFunc(outIn, this.boundSigmoid);
    return outOut;
  }

  matrixFunc(matrix: Matrix, f: (n: number, ...params: number[]) => number, fParams: number[] = []): Matrix {
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

  timesByRand(x: number, offset: number, maxValue: number): number {
    return x * Math.random() * maxValue + offset;
  }

  sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }
}

const neuralNetwork = new NeuralNetwork(3, 3, 3, 3);
console.log(neuralNetwork.query(new Matrix([[1], [0.5], [-1.5]])));
