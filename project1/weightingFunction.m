function weight = weightingFunction()
    weight = [0:1:255];
    weight = min(weight, 255-weight);
end