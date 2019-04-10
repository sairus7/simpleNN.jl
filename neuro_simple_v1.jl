# original version with operations changed to support Julia 1.1,
# + some minor changes to variable naming, data manipulation, etc.

using LinearAlgebra
using MLDatasets
using Random

struct network_v1
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

function (net::network_v1)(a)
    for (w, b) in zip(net.weights, net.biases)
        a = σ.(w*a + b)
    end
    return a
end

function network_v1(sizes)
    num_layers = length(sizes)
    sizearr = sizes
    biases = [randn(y) for y in sizes[2:end]]
    weights = [randn(y, x) for (x, y) in zip(sizes[1:end-1], sizes[2:end])]
    zs = [zeros(y) for y in sizes[2:end]]
    activations = [zeros(y) for y in sizes[1:end]]
    ∇_b = [zeros(y) for y in sizes[2:end]]
    ∇_w = [zeros(y, x) for (x, y) in zip(sizes[1:end-1], sizes[2:end])]
    δ_∇_b = [zeros(y) for y in sizes[2:end]]
    δ_∇_w = [zeros(y, x) for (x, y) in zip(sizes[1:end-1], sizes[2:end])]
    δs = [zeros(y) for y in sizes[2:end]]
    network_v1(num_layers, sizearr, biases, weights, zs, activations,∇_b,∇_w,δ_∇_b,δ_∇_w,δs)
end

function update_batch(net::network_v1, batch, η)
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

    for (x, y) in batch
        backprop!(net, x, y)
        for i in 1:length(∇_b)
            ∇_b[i] .+= δ_∇_b[i]
        end
        for i in 1:length(∇_w)
            ∇_w[i] .+= δ_∇_w[i]
        end
    end

	coef = (η/length(batch))
    for i in 1:length(∇_b)
        net.biases[i] .-= coef.*∇_b[i]
    end

    for i in 1:length(∇_w)
        net.weights[i] .-= coef.*∇_w[i]
    end
end

function backprop!(net::network_v1, x, y)
    ∇_b = net.δ_∇_b
    ∇_w = net.δ_∇_w

    len = net.num_layers - 1
    activations = net.activations
    activations[1] .= x
    zs = net.zs
    δs = net.δs

    for i in 1:len
        b, w, z = net.biases[i], net.weights[i], zs[i]
        mul!(z,w,activations[i]) # z = w * inp
        z .+= b
        activations[i+1] .= σ.(z)
    end

    δ = δs[end]
    δ .= (activations[end] .- y) .* σ_grad.(zs[end])
    ∇_b[end] .= δ

    mul!(∇_w[end], δ, activations[end-1]') # ∇_w[end] = δ * activations[end-1]'

    for l in 1:len-1
        z = zs[end-l]
        mul!(δs[end-l], net.weights[end-l+1]', δ) # δs[end-l] = net.weights[end-l+1]' * δ
        δ = δs[end-l]
        δ .*= σ_grad.(z)
        ∇_b[end-l] .= δ
        mul!(∇_w[end-l], δ, activations[end-l-1]') # ∇_w[end-l] = δ * activations[end-1-l]'
    end
    return nothing
end

function SGDtrain(net::network_v1, traindata, epochs, batch_size, η, testdata=nothing)
    n_test = testdata != nothing ? length(testdata) : nothing
    n = length(traindata)

    traindata = shuffle(traindata) # one time shuffle for performance, then only take random batch index

    println("========")
    for j in 1:epochs
        idx = randperm(n ÷ batch_size) .* batch_size

        batches = [traindata[k-batch_size+1 : k] for k in idx]

        @time for batch in batches
            update_batch(net, batch, η)
        end

        if testdata != nothing
            println("Epoch ", j,": ", evaluate(net, testdata), "/", n_test)
        else
            println("Epoch ", j," complete.")
        end
    end
end


function evaluate(net::network_v1, testdata)
    test_results = [(findmax(net(x))[2] - 1, y) for (x, y) in testdata]
    return sum(Int(x == y) for (x, y) in test_results)
end

function loaddata(rng = 1:60000)
    train_x, train_y = MNIST.traindata(Float64, Vector(rng))
    train_x = [train_x[:,:,x][:] for x in 1:size(train_x, 3)]
    train_y = [vectorize(x) for x in train_y]
    traindata = [(x, y) for (x, y) in zip(train_x, train_y)]

    test_x, test_y = MNIST.testdata(Float64)
    test_x = [test_x[:,:,x][:] for x in 1:size(test_x, 3)]
    testdata = [(x, y) for (x, y) in zip(test_x, test_y)]
    return traindata, testdata
end

function vectorize(n)
    ev = zeros(10)
    ev[n+1] = 1
    return ev
end

function main_v1()
    epochs = 10
    batch_size = 10
    η = 1.25
    net = network_v1([784, 30, 10])
    traindata, testdata = loaddata()
    SGDtrain(net, traindata, epochs, batch_size, η, testdata)
	# @profiler SGDtrain(net, traindata, 1, batch_size, η, testdata)
end
