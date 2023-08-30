class Matrix {
  _matrix;

  constructor(matrix) {
    this._matrix = matrix.map((row) => row.map(Number));
  }

  get shape() {
    return [this._matrix.length, this._matrix[0].length];
  }

  sliceRows(i1, i2) {
    return new Matrix(this._matrix.slice(i1, i2));
  }

  sliceColumns(j1, j2) {
    return new Matrix(this._matrix.map((row) => row.slice(j1, j2)));
  }

  slice(i1, i2, j1, j2) {
    return new Matrix(
      this._matrix.map((row) => row.slice(j1, j2)).slice(i1, i2),
    );
  }

  static block(blocks) {
    const result = [];
    blocks.forEach((blockRow) => {
      for (let i = 0; i < blockRow[0]._matrix.length; i++) {
        result.push(blockRow.map((block) => block._matrix[i]).flat());
      }
    });
    return new Matrix(result);
  }

  static zeros(m, n) {
    return new Matrix([...Array(m)].map(() => Array(n).fill(0)));
  }

  static identity(n) {
    const result = Matrix.zeros(n, n);
    for (let i = 0; i < n; i++) {
      result._matrix[i][i] = 1;
    }
    return result;
  }

  scalarMultiply(k) {
    return new Matrix(this._matrix.map((row) => row.map((x) => x * k)));
  }

  subtract(other) {
    const [m, n] = this.shape;
    const [p, q] = other.shape;

    if (m !== p || n !== q) {
      throw new Error("Subtraction error: Input size mismatch");
    }

    return new Matrix(
      this._matrix.map((row, i) => row.map((x, j) => x - other._matrix[i][j])),
    );
  }

  multiply(other) {
    const [m, n] = this.shape;
    const [p, q] = other.shape;

    if (n !== p) {
      throw new Error("Multiplication error: Input size mismatch");
    }

    const data = [...Array(m)].map((_, i) =>
      [...Array(q)].map((_, j) => {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += this._matrix[i][k] * other._matrix[k][j];
        }
        return sum;
      }),
    );
    return new Matrix(data);
  }

  columnArgmax(j) {
    return this._matrix
      .map((row) => row[j])
      .reduce(
        (argmax, currentValue, currentIndex, array) =>
          Math.abs(currentValue) > Math.abs(array[argmax])
            ? currentIndex
            : argmax,
        0,
      );
  }

  static luDecomp(a) {
    let l, u;
    const [m, n] = a.shape;
    if (m == 1) {
      l = new Matrix([[1]]);
      u = new Matrix(a._matrix);
      return [l, u];
    }

    const a11 = new Matrix([[a._matrix[0][0]]]);
    const a12 = a.slice(0, 1, 1, n);
    const a21 = a.slice(1, m, 0, 1);
    const a22 = a.slice(1, m, 1, n);

    const l11 = new Matrix([[1]]);
    const u11 = a11;

    const l12 = Matrix.zeros(1, n - 1);
    const u12 = a12;

    const l21 = a21.scalarMultiply(1 / u11._matrix[0][0]);
    const u21 = Matrix.zeros(m - 1, 1);

    const s22 = a22.subtract(l21.multiply(u12));
    const [l22, u22] = Matrix.luDecomp(s22);

    l = Matrix.block([
      [l11, l12],
      [l21, l22],
    ]);
    u = Matrix.block([
      [u11, u12],
      [u21, u22],
    ]);
    return [l, u];
  }

  static lupDecomp(a) {
    let l, u, p;
    const [m, n] = a.shape;
    if (m == 1) {
      l = new Matrix([[1]]);
      u = new Matrix(a._matrix);
      p = new Matrix([[1]]);
      return [l, u, p];
    }

    const i = a.columnArgmax(0);
    const aBar = Matrix.block([
      [a.sliceRows(i, i + 1)],
      [a.sliceRows(0, i)],
      [a.sliceRows(i + 1, m)],
    ]);

    const aBar11Scalar = aBar._matrix[0][0];
    const aBar11 = new Matrix([[aBar11Scalar]]);
    const aBar12 = aBar.slice(0, 1, 1, n);
    const aBar21 = aBar.slice(1, m, 0, 1);
    const aBar22 = aBar.slice(1, m, 1, n);

    const s22 = aBar22.subtract(
      aBar21.multiply(aBar12).scalarMultiply(1 / aBar11Scalar),
    );

    const [l22, u22, p22] = Matrix.lupDecomp(s22);

    const l11 = new Matrix([[1]]);
    const u11 = aBar11;

    const l12 = Matrix.zeros(1, m - 1);
    const u12 = aBar12;

    const l21 = p22.multiply(aBar21).scalarMultiply(1 / aBar11Scalar);
    const u21 = Matrix.zeros(m - 1, 1);

    l = Matrix.block([
      [l11, l12],
      [l21, l22],
    ]);
    u = Matrix.block([
      [u11, u12],
      [u21, u22],
    ]);
    const p2 = Matrix.block([
      [new Matrix([[1]]), Matrix.zeros(1, m - 1)],
      [Matrix.zeros(m - 1, 1), p22],
    ]);
    p = Matrix.block([
      [
        p2.sliceColumns(i, i + 1),
        p2.sliceColumns(0, i),
        p2.sliceColumns(i + 1, m),
      ],
    ]);
    return [l, u, p];
  }

  static forwardSub(l, b) {
    const [n, p] = b.shape;
    const x = Matrix.zeros(n, p);
    x._matrix[0] = b
      .sliceRows(0, 1)
      .scalarMultiply(1 / l._matrix[0][0])._matrix[0];
    for (let i = 1; i < n; i++) {
      x._matrix[i] = b
        .sliceRows(i, i + 1)
        .subtract(l.slice(i, i + 1, 0, i).multiply(x.slice(0, i, 0, p)))
        .scalarMultiply(1 / l._matrix[i][i])._matrix[0];
    }
    return x;
  }

  static backwardSub(u, b) {
    const [n, p] = b.shape;
    const x = Matrix.zeros(n, p);
    x._matrix[n - 1] = b
      .sliceRows(n - 1, n)
      .scalarMultiply(1 / u._matrix[n - 1][n - 1])._matrix[0];
    for (let i = n - 2; i >= 0; i--) {
      x._matrix[i] = b
        .sliceRows(i, i + 1)
        .subtract(u.slice(i, i + 1, i + 1, n).multiply(x.slice(i + 1, n, 0, p)))
        .scalarMultiply(1 / u._matrix[i][i])._matrix[0];
    }
    return x;
  }

  static luSolve(l, u, b) {
    const y = Matrix.forwardSub(l, b);
    const x = Matrix.backwardSub(u, y);
    return x;
  }

  static lupSolve(l, u, p, b) {
    const z = p.multiply(b);
    const x = Matrix.luSolve(l, u, z);
    return x;
  }

  static solve(a, b) {
    const [l, u, p] = Matrix.lupDecomp(a);
    const x = Matrix.lupSolve(l, u, p, b);
    return x;
  }

  inverse() {
    const [m, n] = this.shape;

    if (m !== n) {
      throw new Error("Inverse error: Non-square matrix");
    }

    const b = Matrix.identity(m);
    const inv = Matrix.solve(this, b);
    return inv;
  }

  toString() {
    const maxEntryWidth = (array) =>
      Math.max(
        ...array.map((row) => Math.max(...row.map((x) => x.toString().length))),
      );
    const width = maxEntryWidth(this._matrix);

    const [m] = this.shape;

    return this._matrix
      .map(
        (row, i) =>
          `${i === 0 ? "[" : " "}[${row
            .map((x) => x.toString().padStart(width))
            .join(", ")}]${i === m - 1 ? "]" : ""}`,
      )
      .join("\n");
  }
}

function main() {
  a = new Matrix([
    [2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2],
  ]);
  a_inv = a.inverse();
  console.log(a_inv.toString(), "\n"); // [[3/4, 1/2, 1/4], [1/2, 1, 1/2], [1/4, 1/2, 3/4]]

  a = new Matrix([
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2],
  ]);
  b = new Matrix([[8], [-11], [-3]]);
  x = Matrix.solve(a, b); // [[2], [3], [-1]]
  console.log(x.toString(), "\n");
}

main();
