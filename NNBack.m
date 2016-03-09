classdef NNBack < handle
    %NNBack: Neural Network Backpropagation
    % 2 layers: 1 hidden, and 1 output.
    %   Use gradient descent with momentum.
    %   usage: nnet = NNBack(dataInput,numberOfHIddenNodes,numberOfOutPutNodes)
    %
    properties(Access = public)
        % input, hidden, output node
        nIn; 
        nHid;
        nOut;
        % vector of default activation
        aIn;
        aHid;
        aOut;
        %Weight, set to default random[-1,1] if users don't initialize weight
        wIn;
        wOut;
        %%
        %Add momemtum
        momIn;
        momOut;
        
       %------------------------------
       dfIn; % input array, with column is attributes;
    end
    %%
    %             STATIC METHODS                         %
    methods(Static)

       function y = sigmoid(x)
        % Sigmoid function
        % f(x) = 1 / (1+exp(-x))
            y = 1 / (1+exp(-x));
        end
        
        function do = derSigmoid(y)
            %Derivative of sigmoid function f(x) = 1 / (1+exp(-x))
            do = y*(1-y);
        end
    end
    %%
    %                      CTOR                           %
    methods(Access = public)
        % ctor for NNBack class
        function obj = NNBack(dfIn_, nHid_, nOut_,a,b)
            % Contructor of NNBAck.
            % dfIn: a array of dataset INput
            % nHid_: number of hidden node specified by user
            % nOut_: number of output node spceified by user
            % a: lower value of range [a,b]
            % b: upper value of range [a,b]
            obj.dfIn = dfIn_;
            [r,c] = size(dfIn_);
            obj.nIn = c + 1;% one extra node for bias
            obj.nHid = nHid_;
            obj.nOut = nOut_;
            %------------------------------
            % vector of default activation values
            obj.aIn = ones(1,obj.nIn);
            obj.aHid = ones(1,obj.nHid);
            obj.aOut = ones(1,obj.nOut);
            %------------------------------
            if nargin <4 
            %Weight, set to default random[-1,1] if users don't initialize weight
                obj.wIn  = rand(obj.nIn, obj.nHid)*2 -1;
                obj.wOut = rand(obj.nHid, obj.nOut)*2 -1;
            else
                if(b > a)
                    obj.wIn  = rand(obj.nIn, obj.nHid)* (b - a) - a;
                    obj.wOut = rand(obj.nHid, obj.nOut)*(b - a) - a;
                else
                    obj.wIn  = rand(obj.nIn, obj.nHid) *(a - b) - b;
                    obj.wOut = rand(obj.nHid, obj.nOut)*(a - b) - b;
                end
            end
            
            %------------------------------
            %Add momemtum
            obj.momIn = zeros(obj.nIn, obj.nHid);
            obj.momOut = zeros(obj.nHid, obj.nOut);
        end % END CTOR
    end %END PUBLIC METHOD

    %%
    %             PRIVATE METHODS                         %
    methods(Access = private)
        %% feedForward function
        function feedForward(obj,rowIn)
            % activate function in network
            % obj: is instance of NNBack class
            % rowInt: sample input data.(1 row at a time)
            
            % SHOULD VALIDATE THE INPUT ROWS
            % get input activation
            obj.aIn = [rowIn, obj.aIn(end)];
            % caculate the net of input to hidden
            for i=1 : obj.nHid
                netHid = obj.aIn * obj.wIn(:,i);
                obj.aHid(i) = NNBack.sigmoid(netHid);
            end
            
            % caculate the net of hiddent to output
            for i=1: obj.nOut
                netOut = obj.aHid*obj.wOut(:,i);
                obj.aOut(i) = NNBack.sigmoid(netOut);
            end
        end 
        %%
        function error = backProp(obj,target, eta, alpha)
            % Propagate output back to the network and update weight 
            % obj: instance of NNBack class
            % target: an sample target data (1 row at a time)
            % eta: learning rate
            % alpha: momentum rate
            % Return RMSE
            outputE = zeros(1,obj.nOut); % initilize out put error 
            % Backprop from error in output
            for i=1 : obj.nOut
               error = target(i) - obj.aOut(i);
               outputE(i) = NNBack.derSigmoid(obj.aOut(i)) * error;
            end
            
            % Streamdown the network
            % hidden
            hiddenE = zeros(1,obj.nHid);
            for i=1 : obj.nHid
                for j = 1 : obj.nOut
                    error = sum( outputE(j)*obj.wOut(i,:) );
                    hiddenE(i) = NNBack.derSigmoid(obj.aHid(i)) * error;
                end
            end
            
            %Update weight: output to hidden layer
            for i=1:obj.nHid
                for j = 1: obj.nOut
                    wDelta = outputE(j)* obj.aHid(i);
                    obj.wOut(i,j) = obj.wOut(i,j) + eta*wDelta + alpha*obj.momOut(i,j);
                    obj.momOut(i,j) = wDelta;
                end
            end
            
            %Update weight: hidden layer to inputs
            for i=1:obj.nIn
                for j=1: obj.nHid
                    wDelta = hiddenE(j) * obj.aIn(i);
                    obj.wIn(i,j) = obj.wIn(i,j) + eta*wDelta + alpha* obj.momIn(i,j);
                    obj.momIn(i,j) = wDelta;
                end
            end
            [k,c] = size(obj.aOut);
            mse = sum( ((target - obj.aOut).^2)/k);
            rmse = sqrt(mse);
            error = rmse;
        end % end backProp function
        

    end % END PRIVATE METHODS
 
    %%
    %                     PUBLIC                          %
    methods (Access = public)
        %%
        function trainNet(obj, rowTarget, eta, alpha,showPlot, isCV , epochs, minGrad)
            % function to train network, need the target data
            % obj: instance of NNBack class
            % rowTarget: target output array
            % eta: learing rate
            % alpha: momentum rate
            % showPlot: logical value to show plot when the trainning ends.
            % Default is 1.
            % isCV: logical value. 1 to cross validate, 0 if use entire
            % data set. Default is 0.
            % epochs: number of interations. Default is 1000.
            % minGrad: minimum error to terminate the training. Default is
            % 1e-4
            [r,c] = size(obj.dfIn);
            train = ones(r,1);
            test = ones(r,1);
            if nargin < 7
                minGrad = 1e-4;
                epochs = 1000;
                isCV = 0;
                showPlot = 1;
            end
            
            if isCV
                [tr,te] = crossvalind('HoldOut', r ,0.2);
            end
            train = tr;
            test = te;
            displayArr = zeros(2,epochs); % array to plot if needed
            trainInput = obj.dfIn(train,:);
            targetInput = rowTarget(train,:);
            [rTrain,cTrain] = size(trainInput); % number of rows in training set
            size(trainInput)
            for i=1 : epochs 
                error = 0.00;
                errorArr = zeros(1,rTrain);
                for j=1: rTrain
             %       disp(j);
                    obj.feedForward(trainInput(j,:));
                    backPropE = obj.backProp(targetInput(j,:), eta, alpha);
                    errorArr(j) = backPropE;
                    error = error + backPropE;
                end
                displayArr(1,i) = i;
                displayArr(2,i) = error;
                fprintf('Epoch: %d\tmaxRMSE: %6.4f\taveRMSE: %6.4f\n',i,max(errorArr),mean(errorArr));
                if error <= minGrad
                        break;
                end
            end
            
            if showPlot
                x = displayArr(1,:);
                y = displayArr(2,:);
                y = log(y);
                figure;
                plot(x,y);
                xlabel('Epochs');
                ylabel('Performance');
            end
               
        end % end trainNet funciton
        %%
        function predictNet(obj, rowsInput)
            %predict base one weight. Display output value. 
            %obj: instance of NNBack class
            %rowsInput = array of input data to predict.
            [r,c] = size(rowsInput);
            for i=1:r
                obj.feedForward(rowsInput(i,:));
                disp(obj.aOut);
            end
        end
    end % END PUBLIC METHODS
    
end

