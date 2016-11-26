const synaptic = require('synaptic');
const Neuron = synaptic.Neuron,
    Layer = synaptic.Layer,
    Network = synaptic.Network,
    Trainer = synaptic.Trainer,
    Architect = synaptic.Architect;

const data = require("./data.json");
function toVariable(item) {
    return [item.value.just_hilary_percent, item.value.just_hilary_sentiment_score_sentistrength_avg, item.value.just_trump_precent, item.value.just_trump_sentiment_score_sentistrength_avg];
}
function Perceptron(input, hidden, output) {
    // create the layers
    var inputLayer = new Layer(input);
    var hiddenLayer1 = new Layer(hidden);
    var hiddenLayer2 = new Layer(hidden);
    var hiddenLayer3 = new Layer(hidden);

    var outputLayer = new Layer(output);

    // connect the layers
    inputLayer.project(hiddenLayer1);
    hiddenLayer1.project(hiddenLayer2);
    hiddenLayer2.project(hiddenLayer3);

    hiddenLayer3.project(outputLayer);

    // set the layers
    this.set({
        input: inputLayer,
        hidden: [hiddenLayer1, hiddenLayer2, hiddenLayer3],
        output: outputLayer
    });
}
Perceptron.prototype = new Network();
Perceptron.prototype.constructor = Perceptron;

var myPerceptron = new Perceptron(4, 3, 4);
var trainingSet = data.map(function (item) {
    return {
        input: toVariable(item),
        output: toVariable(item),
    };
});

var myTrainer = new Trainer(myPerceptron);
myTrainer.train(trainingSet, {
    rate: 0.0001,
    iterations: 4000000,
    error: .000005,
    shuffle: true,
    log: 10000,
    cost: Trainer.cost.CROSS_ENTROPY
});

var outlierFactor = function (dataIth, oJth, n) {
    return (1 / n) * dataIth.reduce(function (current, actual, n) {
        return current + Math.pow(actual - oJth[n], 2);
    }, 0);
};

var outlierSearch = data.map(function (item) {
    var itemAsVariables = toVariable(item);
    var predicted = myPerceptron.activate(itemAsVariables);
    return { _id: item._id, v: outlierFactor(predicted, itemAsVariables, 4), value: item.value, predicted: predicted,variables: itemAsVariables };
});

console.log(outlierSearch.filter( _ =>  _.v > 0.2 ));

obj = [0.157462686567164, -0.691943127962085, 0.108208955223881, 0.441379310344828];
processed = myPerceptron.activate(obj);
console.log(outlierFactor(obj, processed, 4));