import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = null;

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
};

// normalize continuous values (age, price) to [0, 1] range
// why ? keeps all features balanced so no one dominates training
// formula: (value - min) / (max - min)
// exemple: price=129.99, minPrice=89.99, maxPrice=199.99
// (129.99 - 89.99) / (199.99 - 89.99) = 0.5

const normalize = (value, min, max) => (value - min) / ((max - min) || 1);

function makeContext(products, users) {
    const ages = users.map(u => u.age);
    const prices = products.map(p => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colors = [...new Set(products.map(p => p.color))];
    const categories = [...new Set(products.map(p => p.category))];

    const colorsIndex = Object.fromEntries(
    colors.map((color, index) => [color, index])
    );

    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => [category, index])
    );

    // compute average age
    // help to personalize
    const midAge = (minAge + maxAge) / 2;
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        })
    });

    const productAvgAgeNorm = Object.fromEntries(
        products.map(p => {
            const avg = ageCounts[p.name] ?
            ageSums[p.name] / ageCounts[p.name] : midAge;

            return [p.name, normalize(avg, minAge, maxAge)];
        })
    )

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        productAvgAgeNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // price + age + category + color
        dimension: 2 + colors.length + categories.length
    }
    }

    const oneHotWeighted = (index, length, weight) => tf.oneHot(index, length).cast('float32').mul(weight);

function encodeProduct(product, context) {
    const price = tf.tensor1d([
        normalize(
            product.price,
            context.minPrice,
            context.maxPrice
        ) * WEIGHTS.price
    ]);

    const age = tf.tensor1d([
        (context.productAvgAgeNorm[product.name] ?? 0.5)
        * WEIGHTS.age
    ]);

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    );

    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    );

    return tf.concat([
        price,
        age,
        category,
        color
    ]);
}


function encodeUser(user, context) {
    if (user.purchases?.length) {
        return tf.stack(
            user.purchases.map(p => encodeProduct(p, context))
        )
            .mean(0)
            .reshape([1, context.dimension]);
    }

    const price = tf.zeros([1]);

    const age = tf.tensor1d([
        normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age
    ]);

    const categories = tf.zeros([context.numCategories]);
    const colors = tf.zeros([context.numColors]);

    return tf.concat1d([
        price,
        age,
        categories,
        colors
    ]).reshape([1, context.dimension]);
}

function createTrainingData(context) {
    const inputs = [];
    const labels = [];

    context.users.filter(u => u.purchases.length).forEach(user => {
        // Skip users without purchases
        if (!user.purchases?.length) return;

        const userVector = encodeUser(user, context).dataSync();

        context.products.forEach(product => {
            const productVector = encodeProduct(product, context).dataSync();

            // Check whether the user purchased this product
            const label = user.purchases.some(
                purchase => purchase.name === product.name
            ) ? 1 : 0;

            // Combine user and product features
            inputs.push([...userVector, ...productVector]);
            labels.push(label);
        });
    });

    return {
        inputs: tf.tensor2d(inputs),
        labels: tf.tensor2d(labels, [labels.length, 1]),
        dimension: context.dimension * 2
    };
}


async function configureNeuralNetAndTrain(trainData) {
    const model = tf.sequential();

    model.add(
        tf.layers.dense({
            inputShape: [trainData.dimension],
            units: 128,
            activation: 'relu'
        })
    );

    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    );

    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    );

    model.add(
        tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        })
    );

    model.compile({
        loss: 'binaryCrossentropy',
        optimizer: tf.train.adam(0.001),
        metrics: ['accuracy']
    });

    await model.fit(trainData.inputs, trainData.labels, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch + 1,
                    loss: logs.loss,
                    accuracy: logs.acc ?? logs.accuracy
                });

                postMessage({
                    type: workerEvents.progressUpdate,
                    progress: {
                        progress: Math.round(((epoch + 1) / 100) * 100)
                    }
                });
            }
        }
    });

    return model;
}


async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    const response = await fetch('/data/products.json');

    if (!response.ok) {
        throw new Error(`Failed to load products: ${response.status}`);
    }

    const products = await response.json();

    const context = makeContext(products, users);

    context.productVectors = products.map(p => {
       return {
        name: p.name,
        meta: {...p},
        vector: encodeProduct(p, context).dataSync()
       } 
    });

    _globalCtx = context;

    const trainData = createTrainingData(context);

    _model = await configureNeuralNetAndTrain(trainData);

    trainData.inputs.dispose();
    trainData.labels.dispose();

    postMessage({
        type: workerEvents.trainingComplete
    });
}

function recommend(user) {
    if (!_model) return;
    const context = _globalCtx;

    const userVector = encodeUser(user, _globalCtx).dataSync();

    const inputs = context.productVectors.map(({ vector }) => {
        return [...userVector, ...vector]; // normally use database vector
    });

    const inputTensor = tf.tensor2d(inputs);
    const predictions = _model.predict(inputTensor);

    const scores = predictions.dataSync();
    const recommendations = context.productVectors.map((item, index) => {
        return {
            ...item.meta,
            name: item.name,
            score: scores[index] // predicted score for this product
        }
    })

    const sortedItems = recommendations.sort((a, b) => b.score - a.score);
    
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });
}

const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
