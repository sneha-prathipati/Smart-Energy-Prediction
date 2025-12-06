classdef SmartEnergyApp < matlab.apps.AppBase

    properties (Access = public)
        UIFigure                      matlab.ui.Figure
        EnergyPredictionPanel         matlab.ui.container.Panel
        PredictionOutputLabel         matlab.ui.control.Label
        PredictEnergyButton           matlab.ui.control.Button
        OccupancyFieldEditField       matlab.ui.control.NumericEditField
        OccupancyFieldEditFieldLabel  matlab.ui.control.Label
        Occupancy01Label              matlab.ui.control.Label
        LightFieldEditField           matlab.ui.control.NumericEditField
        LightFieldEditFieldLabel      matlab.ui.control.Label
        LightluxLabel                 matlab.ui.control.Label
        CO2FieldEditField             matlab.ui.control.NumericEditField
        CO2FieldEditFieldLabel        matlab.ui.control.Label
        CO2ppmLabel                   matlab.ui.control.Label
        HumidityFieldEditField        matlab.ui.control.NumericEditField
        HumidityFieldEditFieldLabel   matlab.ui.control.Label
        HumidityLabel                 matlab.ui.control.Label
        TempFieldEditField            matlab.ui.control.NumericEditField
        TempFieldEditFieldLabel       matlab.ui.control.Label
        TemperatureCLabel             matlab.ui.control.Label
        TrainModelsButton             matlab.ui.control.Button
        LoadDataButton                matlab.ui.control.Button
        SmartBuildingEnergyManagementSystemLabel  matlab.ui.control.Label
        UIAxes2                       matlab.ui.control.UIAxes
        UIAxes                        matlab.ui.control.UIAxes
    end

    
    properties (Access = public)
    Data
    ModelRF
    ModelLin
    ModelSVM
    ModelTree
    ModelBoost
    Xtrain
    Ytrain
    Xtest
    Ytest
    end
    

    methods (Access = private)

        function LoadDataButtonPushed(app, event)
[file, path] = uigetfile('*.csv', 'Select Smart Building Data File');
if isequal(file, 0)
    return;
end

data = readtable(fullfile(path, file));

app.Data = data;

uialert(app.UIFigure, 'Data loaded successfully!', 'Success');
app.PredictionOutputLabel.Text = sprintf('Loaded file: %s', file);


        end

        function TrainModelsButtonPushed(app, event)
 
    if ~isprop(app,'Data') || isempty(app.Data)
        uialert(app.UIFigure, 'Please load data first.', 'Error');
        return;
    end

    app.PredictionOutputLabel.Text = 'Training models...';
    drawnow;

    try
        X = app.Data{:, {'Temperature', 'Humidity', 'CO2', 'Light', 'Occupancy'}};
        Y = app.Data.EnergyUsage;

        cv = cvpartition(length(Y), 'HoldOut', 0.3);
        idxTrain = training(cv);
        idxTest  = test(cv);

        app.Xtrain = X(idxTrain,:);
        app.Ytrain = Y(idxTrain);
        app.Xtest  = X(idxTest,:);
        app.Ytest  = Y(idxTest);

        app.ModelRF = TreeBagger(100, app.Xtrain, app.Ytrain, ...
            'Method', 'regression', 'OOBPredictorImportance', 'on', 'OOBPrediction','on');

        app.ModelLin = fitlm(app.Xtrain, app.Ytrain);

        app.ModelSVM = fitrsvm(app.Xtrain, app.Ytrain);

        app.ModelTree = fitrtree(app.Xtrain, app.Ytrain);

        app.ModelBoost = fitrensemble(app.Xtrain, app.Ytrain, 'Method','LSBoost');

        Y_RF    = predict(app.ModelRF, app.Xtest);
        if iscell(Y_RF), Y_RF = str2double(Y_RF); end
        Y_RF = double(Y_RF);

        Y_lin   = predict(app.ModelLin, app.Xtest);
        Y_svm   = predict(app.ModelSVM, app.Xtest);
        Y_tree  = predict(app.ModelTree, app.Xtest);
        Y_boost = predict(app.ModelBoost, app.Xtest);

      safeR2 = @(y,yhat) 1 - sum((y - yhat).^2) / max(sum((y - mean(y)).^2), eps);

        R_RF    = safeR2(app.Ytest, Y_RF);
        R_lin   = safeR2(app.Ytest, Y_lin);
        R_svm   = safeR2(app.Ytest, Y_svm);
        R_tree  = safeR2(app.Ytest, Y_tree);
        R_boost = safeR2(app.Ytest, Y_boost);

        lines = { sprintf('Model R² (test):'), ...
                  sprintf('RandomForest: %.4f', R_RF), ...
                  sprintf('Linear:       %.4f', R_lin), ...
                  sprintf('SVR:          %.4f', R_svm), ...
                  sprintf('DecisionTree: %.4f', R_tree), ...
                  sprintf('Boosted:      %.4f', R_boost) };

        app.PredictionOutputLabel.Text = lines;

        Rvals = [R_RF, R_lin, R_svm, R_tree, R_boost];
        Rvals(isnan(Rvals)) = -Inf;
        [~, bestIdx] = max(Rvals);
        switch bestIdx
            case 1, bestPred = Y_RF; bestName = 'Random Forest';
            case 2, bestPred = Y_lin; bestName = 'Linear Regression';
            case 3, bestPred = Y_svm; bestName = 'SVR';
            case 4, bestPred = Y_tree; bestName = 'Decision Tree';
            case 5, bestPred = Y_boost; bestName = 'Boosted Ensemble';
            otherwise, bestPred = Y_RF; bestName = 'Random Forest';
        end

        cla(app.UIAxes);
        plot(app.UIAxes, app.Ytest, '-o', 'LineWidth', 1.5); hold(app.UIAxes, 'on');
        plot(app.UIAxes, bestPred, '-*', 'LineWidth', 1.2); hold(app.UIAxes, 'off');
        legend(app.UIAxes, {'Actual','Predicted (best)'}, 'Location','best');
        title(app.UIAxes, sprintf('Actual vs Predicted (%s)', bestName));
        xlabel(app.UIAxes,'Sample'); ylabel(app.UIAxes,'Energy Usage'); grid(app.UIAxes,'on');

        if isprop(app,'ModelRF') && ~isempty(app.ModelRF)
            imp = app.ModelRF.OOBPermutedPredictorDeltaError;
            cla(app.UIAxes2);
            bar(app.UIAxes2, imp);
            app.UIAxes2.XTickLabel = {'Temperature','Humidity','CO2','Light','Occupancy'};
            title(app.UIAxes2,'Feature Importance (RF)');
            ylabel(app.UIAxes2,'Importance Score');
            grid(app.UIAxes2,'on');
        end

    catch ME
        uialert(app.UIFigure, sprintf('Training failed:\n%s', ME.message), 'Training Error');
        app.PredictionOutputLabel.Text = 'Training failed. See alert.';
        return;
    end


        end

        function PredictEnergyButtonPushed(app, event)
    if ~isprop(app,'ModelRF') || isempty(app.ModelRF)
        uialert(app.UIFigure, 'Please train the models first.', 'No Model Available');
        return;
    end

    try
        temp  = app.TempFieldEditField.Value;
        hum   = app.HumidityFieldEditField.Value;
        co2   = app.CO2FieldEditField.Value;
        light = app.LightFieldEditField.Value;
        occ   = app.OccupancyFieldEditField.Value;

        if any(isnan([temp hum co2 light occ]))
            app.PredictionOutputLabel.Text = 'Please fill all input fields.';
            return;
        end

        xnew = [temp, hum, co2, light, occ];

        yhat = predict(app.ModelRF, xnew);
        if iscell(yhat), yhat = str2double(yhat); end
        yhat = double(yhat);

        if isprop(app,'Ytrain') && ~isempty(app.Ytrain)
            baseline = mean(app.Ytrain);
        else
            baseline = mean(app.Data.EnergyUsage);
        end

if yhat > baseline
    msg = sprintf(['Predicted Energy = %.2f\n' ...
                   'Recommendation:\n' ...
                   '- Reduce AC usage\n' ...
                   '- Turn off extra lights\n' ...
                   '- Improve ventilation'], yhat);
else
    msg = sprintf('Predicted Energy = %.2f\nStatus: Normal usage', yhat);
end

uialert(app.UIFigure, msg, 'Prediction Result');


    catch ME
        uialert(app.UIFigure, sprintf('Prediction Error:\n%s', ME.message), 'Error');
        app.PredictionOutputLabel.Text = 'Prediction failed.';
    end
   
        end
    end

    methods (Access = private)

        function createComponents(app)

            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 712 480];
            app.UIFigure.Name = 'MATLAB App';

            app.UIAxes = uiaxes(app.UIFigure);
            title(app.UIAxes, 'Actual vs Predicted')
            xlabel(app.UIAxes, 'Sample')
            ylabel(app.UIAxes, 'Energy Usage')
            zlabel(app.UIAxes, 'Z')
            app.UIAxes.Position = [9 197 304 238];

            app.UIAxes2 = uiaxes(app.UIFigure);
            title(app.UIAxes2, 'Feature Importance')
            ylabel(app.UIAxes2, 'Importance Score')
            zlabel(app.UIAxes2, 'Z')
            app.UIAxes2.Position = [319 197 300 238];

            app.SmartBuildingEnergyManagementSystemLabel = uilabel(app.UIFigure);
            app.SmartBuildingEnergyManagementSystemLabel.HorizontalAlignment = 'center';
            app.SmartBuildingEnergyManagementSystemLabel.FontSize = 24;
            app.SmartBuildingEnergyManagementSystemLabel.FontWeight = 'bold';
            app.SmartBuildingEnergyManagementSystemLabel.Position = [75 491 505 31];
            app.SmartBuildingEnergyManagementSystemLabel.Text = 'Smart Building Energy Management System';

            app.LoadDataButton = uibutton(app.UIFigure, 'push');
            app.LoadDataButton.ButtonPushedFcn = createCallbackFcn(app, @LoadDataButtonPushed, true);
            app.LoadDataButton.Position = [20 451 100 22];
            app.LoadDataButton.Text = 'Load Data';

            app.TrainModelsButton = uibutton(app.UIFigure, 'push');
            app.TrainModelsButton.ButtonPushedFcn = createCallbackFcn(app, @TrainModelsButtonPushed, true);
            app.TrainModelsButton.Position = [136 451 100 22];
            app.TrainModelsButton.Text = 'Train Models';

            app.EnergyPredictionPanel = uipanel(app.UIFigure);
            app.EnergyPredictionPanel.Title = 'Energy Prediction Panel';
            app.EnergyPredictionPanel.Position = [36 1 583 185];

            app.TemperatureCLabel = uilabel(app.EnergyPredictionPanel);
            app.TemperatureCLabel.Position = [15 136 97 22];
            app.TemperatureCLabel.Text = 'Temperature (°C)';

            app.TempFieldEditFieldLabel = uilabel(app.EnergyPredictionPanel);
            app.TempFieldEditFieldLabel.HorizontalAlignment = 'right';
            app.TempFieldEditFieldLabel.Position = [129 137 60 22];
            app.TempFieldEditFieldLabel.Text = 'TempField';

            app.TempFieldEditField = uieditfield(app.EnergyPredictionPanel, 'numeric');
            app.TempFieldEditField.Position = [204 137 100 22];

            app.HumidityLabel = uilabel(app.EnergyPredictionPanel);
            app.HumidityLabel.Position = [15 104 74 22];
            app.HumidityLabel.Text = 'Humidity (%)';

            app.HumidityFieldEditFieldLabel = uilabel(app.EnergyPredictionPanel);
            app.HumidityFieldEditFieldLabel.HorizontalAlignment = 'right';
            app.HumidityFieldEditFieldLabel.Position = [111 105 78 22];
            app.HumidityFieldEditFieldLabel.Text = 'HumidityField';

            app.HumidityFieldEditField = uieditfield(app.EnergyPredictionPanel, 'numeric');
            app.HumidityFieldEditField.Position = [204 105 100 22];

            app.CO2ppmLabel = uilabel(app.EnergyPredictionPanel);
            app.CO2ppmLabel.Position = [15 73 64 22];
            app.CO2ppmLabel.Text = 'CO2 (ppm)';

            app.CO2FieldEditFieldLabel = uilabel(app.EnergyPredictionPanel);
            app.CO2FieldEditFieldLabel.HorizontalAlignment = 'right';
            app.CO2FieldEditFieldLabel.Position = [133 74 56 22];
            app.CO2FieldEditFieldLabel.Text = 'CO2Field';

            app.CO2FieldEditField = uieditfield(app.EnergyPredictionPanel, 'numeric');
            app.CO2FieldEditField.Position = [204 74 100 22];

            app.LightluxLabel = uilabel(app.EnergyPredictionPanel);
            app.LightluxLabel.Position = [15 43 58 22];
            app.LightluxLabel.Text = 'Light (lux)';

            app.LightFieldEditFieldLabel = uilabel(app.EnergyPredictionPanel);
            app.LightFieldEditFieldLabel.HorizontalAlignment = 'right';
            app.LightFieldEditFieldLabel.Position = [132 43 57 22];
            app.LightFieldEditFieldLabel.Text = 'LightField';

            app.LightFieldEditField = uieditfield(app.EnergyPredictionPanel, 'numeric');
            app.LightFieldEditField.Position = [204 43 100 22];

            app.Occupancy01Label = uilabel(app.EnergyPredictionPanel);
            app.Occupancy01Label.Position = [14 13 90 22];
            app.Occupancy01Label.Text = 'Occupancy(0/1)';

            app.OccupancyFieldEditFieldLabel = uilabel(app.EnergyPredictionPanel);
            app.OccupancyFieldEditFieldLabel.HorizontalAlignment = 'right';
            app.OccupancyFieldEditFieldLabel.Position = [98 14 91 22];
            app.OccupancyFieldEditFieldLabel.Text = 'OccupancyField';

            app.OccupancyFieldEditField = uieditfield(app.EnergyPredictionPanel, 'numeric');
            app.OccupancyFieldEditField.Position = [204 14 100 22];

            app.PredictEnergyButton = uibutton(app.EnergyPredictionPanel, 'push');
            app.PredictEnergyButton.ButtonPushedFcn = createCallbackFcn(app, @PredictEnergyButtonPushed, true);
            app.PredictEnergyButton.Position = [336 136 100 22];
            app.PredictEnergyButton.Text = 'Predict Energy';

            app.PredictionOutputLabel = uilabel(app.EnergyPredictionPanel);
            app.PredictionOutputLabel.VerticalAlignment = 'top';
            app.PredictionOutputLabel.WordWrap = 'on';
            app.PredictionOutputLabel.FontSize = 14;
            app.PredictionOutputLabel.Position = [448 14 124 144];
            app.PredictionOutputLabel.Text = {'Prediction'; 'Output'; 'Label'};

            app.UIFigure.Visible = 'on';
        end
    end

    methods (Access = public)

        function app = SmartEnergyApp

            createComponents(app)

            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        function delete(app)

            delete(app.UIFigure)
        end
    end
end
