import { Matrix } from 'ml-matrix';

interface queryState {
  hidIn: Matrix,
  hidOut: Matrix,
  outIn: Matrix,
  outOut: Matrix,
}

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
    const baseInToHid = Matrix.ones(this.hidNodes, this.inNodes); // check hid and in are right way around
    const baseHidToOut = Matrix.ones(this.outNodes, this.hidNodes);

    this.inToHidWeights = this.matrixFunc(baseInToHid, this.boundTimesByRand, [-0.5, 1]);
    this.hidToOutWeights = this.matrixFunc(baseHidToOut, this.boundTimesByRand, [-0.5, 1]);
  }

  train(input: Matrix, target: Matrix): void {
    console.log(input, target);
    const query = this.query(input);
    const outErr = Matrix.sub(target, query.outOut);
    const hidErr = this.hidToOutWeights.mmul(outErr);

    const oneMinOut = Matrix.sub(Matrix.ones(this.outNodes, 1), query.outOut);
    const outMulOneMinOut = query.outOut.mmul(oneMinOut);
    const outTrans = query.outOut.transpose();
    const deltaHidToOut = Matrix.mul(outErr.mmul(outTrans).mmul(outMulOneMinOut), this.learningRate);
    Matrix.add(this.hidToOutWeights, deltaHidToOut);

    const oneMinHidOut = Matrix.sub(Matrix.ones(this.hidNodes, 1), query.hidOut);
    const hidOutMulOneMinHidOut = query.hidOut.mmul(oneMinHidOut);
    const hidOutTrans = query.hidOut.transpose();
    const deltaInToHid = Matrix.mul(hidErr.mmul(hidOutTrans).mmul(hidOutMulOneMinHidOut), this.learningRate);
    Matrix.add(this.inToHidWeights, deltaInToHid);
  }

  query(input: Matrix): queryState {
    // calculate signals into hidden layer, then output of hidden layer
    const hidIn: Matrix = this.inToHidWeights.mmul(input);
    const hidOut: Matrix = this.matrixFunc(hidIn, this.boundSigmoid);

    // calculate signals into output layer, then output of output layer
    const outIn: Matrix = this.hidToOutWeights.mmul(hidOut);
    const outOut: Matrix = this.matrixFunc(outIn, this.boundSigmoid);
    return { hidIn, hidOut, outIn, outOut };
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
