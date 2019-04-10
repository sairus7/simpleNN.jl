# network structure rework: split into 3 different structures: network, batch_trainer and batch_tester for preallocation
# the whole run time is faster due to preallocation for evaluation batch (batch_tester)
using LinearAlgebra
using MLDatasets
using Random

@inline σ(z) = 1/(1+exp(-z))
@inline σ_grad(z) = σ(z)*(1-σ(z))

struct network_v4
    num_layers::Int64
    sizearr::Array{Int64,1}
    biases::Array{Array{Float64,1},1}
    weights::Array{Array{Float64,2},1}
end
function network_v4(sizes)
    num_layers = length(sizes)
    sizearr = sizes
    biases = [randn(y) for y in sizes[2:end]]
    weights = [randn(y, x) for (x, y) in zip(sizes[1:end-1], sizes[2:end])]
    network_v4(num_layers, sizearr, biases, weights)
end
function (net::network_v4)(a)
    for (w, b) in zip(net.weights, net.biases)
        a = σ.(w*a .+ b)
    end
    return a
end

struct batch_trainer
    η::Float64
    batch_size::Int64
    ∇_b::Array{Array{Float64,1},1}
    ∇_w::Array{Array{Float64,2},1}
    zs::Array{Array{Float64,2},1}
    activations::Array{Array{Float64,2},1}
    δs::Array{Array{Float64,2},1}
end
function batch_trainer(net::network_v4, batch_size, η)
    sizes = net.sizearr
    ∇_b = [zeros(y) for y in sizes[2:end]]
    ∇_w = [zeros(y, x) for (x, y) in zip(sizes[1:end-1], sizes[2:end])]
    zs = [zeros(y, batch_size) for y in sizes[2:end]]
    activations = [zeros(y, batch_size) for y in sizes[2:end]]
    δs = [zeros(y, batch_size) for y in sizes[2:end]]
    batch_trainer(η, batch_size, ∇_b, ∇_w, zs, activations, δs)
end

struct batch_tester
    batch_size::Int64
    zs::Array{Array{Float64,2},1}
    activations::Array{Array{Float64,2},1}
    δs::Array{Array{Float64,2},1}
end
function batch_tester(net::network_v4, batch_size)
    sizes = net.sizearr
    zs = [zeros(y, batch_size) for y in sizes[2:end]]
    activations = [zeros(y, batch_size) for y in sizes[2:end]]
    δs = [zeros(y, batch_size) for y in sizes[2:end]]
    batch_tester(batch_size, zs, activations, δs)
end

# forward pass for testing
function (tester::batch_tester)(net::network_v4, x)
    activations = tester.activations
    zs = tester.zs
    len = length(activations)

	input = x
    for i in 1:len
        b, w, z = net.biases[i], net.weights[i], zs[i]
        mul!(z, w, input) # z = w * input
        z .+= b
        activations[i] .= σ.(z)
        input = activations[i]
    end
    return activations[end]
end

# forward and backprop for training
function (trainer::batch_trainer)(net::network_v4, x, y)
    ∇_b = trainer.∇_b
    ∇_w = trainer.∇_w
	len = net.num_layers - 1
    activations = trainer.activations
    zs = trainer.zs
    δs = trainer.δs

    input = x
    for i in 1:len
        b, w, z = net.biases[i], net.weights[i], zs[i]
        mul!(z, w, input) # z = w * input
        z .+= b
        activations[i] .= σ.(z)
        input = activations[i]
    end

    δ = δs[end]
    δ .= (activations[end] .- y) .* σ_grad.(zs[end])
    sum!(∇_b[end], δ)

    for l in 1:len-1
        mul!(∇_w[end-l+1], δ, activations[end-l]') # ∇_w[end-l+1] = δ * activations[end-l]'
        z = zs[end-l]
        mul!(δs[end-l], net.weights[end-l+1]', δ) # δs[end-l] = net.weights[end-l+1]' * δ
        δ = δs[end-l]
        δ .*= σ_grad.(z)
        sum!(∇_b[end-l], δ)
    end
    mul!(∇_w[1], δ, x') # ∇_w[1] = δ * x'

    return nothing
end

function update_batch(net::network_v4, trainer::batch_trainer, x, y)

    trainer(net, x, y)

    coef = trainer.η/size(x,2)
    for i in 1:length(trainer.∇_b)
        net.biases[i] .-= coef .* trainer.∇_b[i]
    end
    for i in 1:length(trainer.∇_w)
        net.weights[i] .-= coef .* trainer.∇_w[i]
    end
end

function SGDtrain(net::network_v4, trainer::batch_trainer, traindata, epochs, tester, testdata=nothing)
    n_test = testdata != nothing ? size(testdata[1], 2) : nothing
    n = size(traindata[1], 2)

    idx = randperm(n) # one time shuffle for performance, then only take random batch index
    # idx = 1:n
    train_x = traindata[1][:,idx]
    train_y = traindata[2][:,idx]
    test_x, test_y = testdata

    # reorganize data in batches
    batch = [(train_x[:, k-batch_size+1 : k], train_y[:, k-batch_size+1 : k]) for k in batch_size:batch_size:n]

    println("========")
    for j in 1:epochs
        idx = randperm(length(batch))

        @time for k in idx
            update_batch(net, trainer, batch[k]...)
        end

        if testdata != nothing
            println("Epoch ", j,": ", evaluate(tester(net, test_x), test_y), "/", tester.batch_size)
        else
            println("Epoch ", j," complete.")
        end
    end
end

function evaluate(out, y)
    hits = 0
    for i = 1:size(out, 2)
        if (findmax(out[:,i])[2] - 1) == y[i]
            hits += 1
        end
    end
    hits
end

function loaddata(rng = 1:60000)
    train_x, train_y = MNIST.traindata(Float64, Vector(rng))
    train_x = reshape(train_x, size(train_x,1)*size(train_x,2), :) # 28 x 28 x N -> 28*28 x N
    train_y = vectorize(train_y)
    test_x, test_y = MNIST.testdata(Float64)
    test_x = reshape(test_x, size(test_x,1)*size(test_x,2), :) # 28 x 28 x N -> 28*28 x N
    return (train_x, train_y), (test_x, test_y)
end

function vectorize(vec)
    N = 10
    len = length(vec)
    mtx = zeros(N, len)
    for i = 1:len
        mtx[vec[i]+1, i] = 1
    end
    return mtx
end

function main_v4()
    epochs = 10
    batch_size = 10
    η = 1.25
    net = network_v4([784, 30, 10])
    traindata, testdata = loaddata()
	trainer = batch_trainer(net, batch_size, η)
	tester = batch_tester(net, size(testdata[1],2))
    SGDtrain(net, trainer, traindata, epochs, tester, testdata)
	# @profiler SGDtrain(net, trainer, traindata, 1, tester, testdata)
end
