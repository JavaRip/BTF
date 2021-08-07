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
    // const baseInToHid = Matrix.ones(this.hidNodes, this.inNodes); // check hid and in are right way around
    // const baseHidToOut = Matrix.ones(this.outNodes, this.hidNodes);

    // this.inToHidWeights = this.matrixFunc(baseInToHid, this.boundTimesByRand, [-0.5, 1]);
    this.inToHidWeights = new Matrix([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]]);
    console.log(this.inToHidWeights);
    // this.hidToOutWeights = this.matrixFunc(baseHidToOut, this.boundTimesByRand, [-0.5, 1]);
    this.hidToOutWeights = new Matrix([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]);
    console.log(this.hidToOutWeights);
    console.log('######################');
  }

  train(input: Matrix, target: Matrix): void {
    const query = this.query(input);
    const outErr = Matrix.sub(target, query.outOut);
    const hidErr = this.hidToOutWeights.transpose().mmul(outErr);

    const oneMinOut = Matrix.sub(Matrix.ones(this.outNodes, 1), query.outOut);
    const outMulOutErr = this.pyMultiply(outErr, query.outOut);
    const minOutMulOutMulErr = this.pyMultiply(oneMinOut, outMulOutErr);
    const deltaHidOut = minOutMulOutMulErr.mmul(query.hidOut.transpose());
    const deltaHidOutMulLr = Matrix.mul(deltaHidOut, this.learningRate);
    this.hidToOutWeights = Matrix.add(deltaHidOutMulLr, this.hidToOutWeights);
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

  // multiplies x,1 matrices as python would
  pyMultiply(a: Matrix, b: Matrix): Matrix {
    const temp: number[][] = [];
    for (let i = 0; i < a.rows; i += 1) {
      temp.push([a.get(i, 0) * b.get(i, 0)]);
    }
    return new Matrix(temp);
  }

  timesByRand(x: number, offset: number, maxValue: number): number {
    return x * Math.random() * maxValue + offset;
  }

  sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }
}

const neuralNetwork = new NeuralNetwork(3, 3, 3, 0.3);
neuralNetwork.query(new Matrix([[1], [0.5], [-1.5]]));
console.log('---------query complete-------------');
neuralNetwork.train(new Matrix([[1], [0.5], [-1.5]]), new Matrix([[1], [2], [3]]));
console.log('---------training complete-------------');
neuralNetwork.query(new Matrix([[1], [0.5], [-1.5]]));
console.log('---------query complete-------------');

