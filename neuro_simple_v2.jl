# input data format changed to batch matrix instead of batch of tuples (loaddata, vectorize, etc.)
# first activation layer removed, no need to copy input data to it

using LinearAlgebra
using MLDatasets
using Random

struct network_v2
    num_layers::Int64
    sizearr::Array{Int64,1}
    biases::Array{Array{Float64,1},1}
    weights::Array{Array{Float64,2},1}
    zs::Array{Array{Float64,1},1}
    activations::Array{Array{Float64,1},1}
    ∇_b::Array{Array{Float64,1},1}
    ∇_w::Array{Array{Float64,2},1}
    δ_∇_b::Array{Array{Float64,1},1}
    δ_∇_w::Array{Array{Float64,2},1}
    δs::Array{Array{Float64,1},1}
end

σ(z) = 1/(1+exp(-z))
σ_grad(z) = σ(z)*(1-σ(z))

function (net::network_v2)(a)
    for (w, b) in zip(net.weights, net.biases)
        a = σ.(w*a + b)
    end
    return a
end

function network_v2(sizes)
    num_layers = length(sizes)
    sizearr = sizes
    biases = [randn(y) for y in sizes[2:end]]
    weights = [randn(y, x) for (x, y) in zip(sizes[1:end-1], sizes[2:end])]
    zs = [zeros(y) for y in sizes[2:end]]
    activations = [zeros(y) for y in sizes[2:end]]
    ∇_b = [zeros(y) for y in sizes[2:end]]
    ∇_w = [zeros(y, x) for (x, y) in zip(sizes[1:end-1], sizes[2:end])]
    δ_∇_b = [zeros(y) for y in sizes[2:end]]
    δ_∇_w = [zeros(y, x) for (x, y) in zip(sizes[1:end-1], sizes[2:end])]
    δs = [zeros(y) for y in sizes[2:end]]
    network_v2(num_layers, sizearr, biases, weights, zs, activations,∇_b,∇_w,δ_∇_b,δ_∇_w,δs)
end

function update_batch(net::network_v2, x, y, η)
    ∇_b = net.∇_b
    ∇_w = net.∇_w

    for i in 1:length(∇_b)
      ∇_b[i] .= 0.0
    end

    for i in 1:length(∇_w)
      ∇_w[i] .= 0.0
    end

    δ_∇_b = net.δ_∇_b
    δ_∇_w = net.δ_∇_w

    for k in 1:size(y,2)
        xk = @view x[:,k]
        yk = @view y[:,k]
        backprop!(net, xk, yk)

        for i in 1:length(∇_b)
            ∇_b[i] .+= δ_∇_b[i]
        end
        for i in 1:length(∇_w)
            ∇_w[i] .+= δ_∇_w[i]
        end
    end

    coef = η/size(x,2)

    for i in 1:length(∇_b)
        net.biases[i] .-= coef.*∇_b[i]
    end

    for i in 1:length(∇_w)
        net.weights[i] .-= coef.*∇_w[i]
    end
end

function backprop!(net::network_v2, x, y)
    ∇_b = net.δ_∇_b
    ∇_w = net.δ_∇_w

    len = net.num_layers - 1
    activations = net.activations
    zs = net.zs
    δs = net.δs

    input = x
    for i in 1:len
        b, w, z = net.biases[i], net.weights[i], zs[i]
        mul!(z, w, input) # z = w * inp
        z .+= b
        activations[i] .= σ.(z)
        input = activations[i]
    end

    δ = δs[end]
    δ .= (activations[end] .- y) .* σ_grad.(zs[end])
    ∇_b[end] .= δ

    for l in 1:len-1
        mul!(∇_w[end-l+1], δ, activations[end-l]') # ∇_w[end-l+1] = δ * activations[end-l]'
        z = zs[end-l]
        mul!(δs[end-l], net.weights[end-l+1]', δ) # δs[end-l] = net.weights[end-l+1]' * δ
        δ = δs[end-l]
        δ .*= σ_grad.(z)
        ∇_b[end-l] .= δ
    end
    mul!(∇_w[1], δ, x') # ∇_w[1] = δ * x'

    return nothing
end

function SGDtrain(net::network_v2, traindata, epochs, batch_size, η, testdata=nothing)
    n_test = testdata != nothing ? size(testdata[1], 2) : nothing
    n = size(traindata[1], 2)

    idx = randperm(n) # one time shuffle for performance, then only take random batch index
    train_x = traindata[1][:,idx]
    train_y = traindata[2][:,idx]

    # reorganize data in batches
    batch = [(train_x[:, k-batch_size+1 : k], train_y[:, k-batch_size+1 : k]) for k in batch_size:batch_size:n]

    println("========")
    for j in 1:epochs
        idx = randperm(length(batch))

        @time for k in idx
            update_batch(net, batch[k]..., η)
        end

        if testdata != nothing
            println("Epoch ", j,": ", evaluate(net, testdata...), "/", n_test)
        else
            println("Epoch ", j," complete.")
        end
    end
end

function evaluate(net::network_v2, x, y)
    hits = 0
    for i = 1:size(x, 2)
        out = net(x[:,i])
        if (findmax(out)[2] - 1) == y[i]
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

function main_v2()
    epochs = 10
    batch_size = 10
    η = 1.25
    net = network_v2([784, 30, 10])
    traindata, testdata = loaddata()
    SGDtrain(net, traindata, epochs, batch_size, η, testdata)
	# @profiler SGDtrain(net, traindata, 1, batch_size, η, testdata)
end
