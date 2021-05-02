const w = 20;
let cols, rows;
let model;
const training_xs = tf.tensor2d([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
]),
    training_ys = tf.tensor2d([
        [0],
        [1],
        [1],
        [0]
    ]);
let xs, y_vals = [];
const losses = [];

async function setup() {
    createCanvas(600, 600);
    cols = width / w;
    rows = height / w;
    //create the data
    let inputs = [];
    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            let x1 = i / cols;
            let x2 = j / rows;
            inputs.push([x1, x2]);
            y_vals.push(0.5);
        }
    }
    xs = tf.tensor2d(inputs);
    //create the model
    model = tf.sequential();
    const layer1 = tf.layers.dense({
        units: 2,
        inputShape: [2],
        activation: 'sigmoid'
    });
    const outputLayer = tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    });

    model.add(layer1);
    model.add(outputLayer);
    const optimizer = tf.train.adam(0.1);
    const compileConfig = {
        optimizer: optimizer,
        loss: 'meanSquaredError'
    };
    model.compile(compileConfig);

    //Training and Extracting the data
    setTimeout(train, 3);
    setInterval(getYs, 300);
}

async function getYs() {
    let ys = model.predict(xs);
    y_vals = await ys.data();
    ys.dispose();
}

async function train() {
    let response = await training();
    //console.log(response.history.loss[0]);
    //console.log(tf.memory().numTensors);
    //console.log(y_vals[0], y_vals[y_vals.length - 1]);
    losses.push(response.history.loss[0]);
    setTimeout(train, 3);
}

function training() {
    const option = {
        shuffle: true,
        epochs: 1
    };
    return model.fit(training_xs,
        training_ys, option);
}

function draw() {
    background(0);

    let index = 0;
    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            let br = y_vals[index] * 255;
            fill(br);
            rect(i * w, j * w, w, w);
            fill(255 - br);
            textSize(8);
            textAlign(CENTER, CENTER);
            text(
                nf(y_vals[index], 1, 2),
                i * w + w / 2,
                j * w + w / 2
            );
            index++;
        }
    }
}