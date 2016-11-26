angular.module('ui.bootstrap.demo', ['ngAnimate', 'ngSanitize', 'ui.bootstrap']);
angular.module('ui.bootstrap.demo').controller('ButtonsCtrl', function ($scope, $http) {
    const Layer = synaptic.Layer,
        Network = synaptic.Network,
        Trainer = synaptic.Trainer,
        Architect = synaptic.Architect;
    $http.get("./data.json").then(function (data) {
        $scope.data = data.data;
        $scope.to_plot = $scope.data;
    });
    function toVariable(item) {
        return [item.value.just_hillary_percent, item.value.just_hillary_sentiment_score_sentistrength_avg, item.value.just_trump_percent, item.value.just_trump_sentiment_score_sentistrength_avg];
    }


    var outlierFactor = function (dataIth, oJth, n) {
        return (1 / n) * dataIth.reduce(function (current, actual, n) {
            return current + Math.pow(actual - oJth[n], 2);
        }, 0);
    };

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

    var perceptron = new Perceptron(4, 3, 4);
    $scope.clear = function (data) {
        data.forEach(item => { item.outlier = false; })
    };

    $scope.applyRnn = function (data, outlinerFactoryValue) {
        $scope.clear(data);
        var trainingSet = data.map(function (item) {
            return {
                input: toVariable(item),
                output: toVariable(item),
            };
        });
        var trainingSet = data.map(function (item) {
            return {
                input: toVariable(item),
                output: toVariable(item),
            };
        });

        var trainer = new Trainer(perceptron);
        trainer.train(trainingSet, {
            rate: 0.0001,
            iterations: 2000,
            error: .000005,
            shuffle: true,
            log: 10000,
            cost: Trainer.cost.CROSS_ENTROPY
        });
        var outlierSearch = data.map(function (item) {
            var itemAsVariables = toVariable(item);
            var predicted = perceptron.activate(itemAsVariables);
            return { _id: item._id, v: outlierFactor(predicted, itemAsVariables, 4), item: item, value: item.value, predicted: predicted, variables: itemAsVariables };
        });
        var outliers = outlierSearch.filter(_ => _.v > outlinerFactoryValue);
        outliers.forEach(item => { item.item.outlier = true; })
        $scope.to_plot = data;
    };

});