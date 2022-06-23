from numpy.random import rand
from numpy.random import randn
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from numpy import ones
from numpy import hstack
from numpy import zeros

# generate randoms sample from x^2
def generate_samples(n=100):
    # generate random inputs in [-0.5, 0.5]
    X1 = rand(n) - 0.5
    # generate outputs X^2 (quadratic)
    X2 = X1 * X1
    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    return hstack((X1, X2))


# generate n real samples with class labels
def generate_real_samples(n):
    # generate inputs in [-0.5, 0.5]
    X1 = rand(n) - 0.5

    # generate outputs X^2
    X2 = X1 * X1

    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))

    # generate class labels
    y = ones((n, 1))

    return X, y


# define the standalone generator model
def define_generator(latent_dim, n_outputs=2):
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(n_outputs, activation='linear'))
    return model


# define the standalone discriminator model
def define_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# train the discriminator model
def train_discriminator(model, n_epochs=1000, n_batch=128):
    half_batch = int(n_batch / 2)

    # run epochs manually
    for i in range(n_epochs):

        # generate real examples
        X_real, y_real = generate_real_samples(half_batch)

        # update model
        model.train_on_batch(X_real, y_real)

        # generate fake examples
        X_fake, y_fake = generate_fake_samples(half_batch)

        # update model
        model.train_on_batch(X_fake, y_fake)

        # evaluate the model
        _, acc_real = model.evaluate(X_real, y_real, verbose=0)
        _, acc_fake = model.evaluate(X_fake, y_fake, verbose=0)

        # print after every 50 ietrations
        if (i + 1) % 50 == 0:
            print(i + 1, acc_real, acc_fake)


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
    # generate points in the latent space
    x_input = randn(latent_dim * n)

    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n)

    # predict outputs
    X = generator.predict(x_input)

    # create class labels
    y = zeros((n, 1))
    return X, y


# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False

    # connect them
    model = Sequential()

    # add generator
    model.add(generator)

    # add the discriminator
    model.add(discriminator)

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000):
    # determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)

    # compile generator
    g_model.compile(loss='binary_crossentropy', optimizer='adam')

    # manually enumerate epochs
    for i in range(n_epochs):

        # prepare real samples
        x_real, y_real = generate_real_samples(half_batch)

        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

        # update discriminator
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)

        # prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)

        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))

        # update the generator via the discriminator's error
        gan_model.train_on_batch(x_gan, y_gan)

        # evaluate the model every n_eval epochs
        if (i + 1) % n_eval == 0:
            summarize_performance(i, g_model, d_model, latent_dim)


# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    # prepare real samples
    x_real, y_real = generate_real_samples(n)

    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)

    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)

    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)

    # prepare points in latent space as input for the generator
    x_gan = generate_latent_points(latent_dim, n)

    # create inverted labels for the fake samples
    y_gan = ones((n, 1))

    # evaluate generator's accuracy and summarize
    acc_gen = generator.evaluate(x_gan, y_gan, verbose=0)
    print(epoch, " - Generator: ", acc_gen)

    # summarize discriminator performance
    print("Discriminator: ", acc_real, acc_fake)
    print()

    # scatter plot real and fake data points
    pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
    pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    pyplot.show()


# generate samples
data = generate_samples()

# plot samples
pyplot.figure(1)
pyplot.scatter(data[:, 0], data[:, 1])
pyplot.show()

# define the discriminator model
discriminator = define_discriminator()

# size of the latent space
latent_dim = 5

# define the discriminator model
generator = define_generator(latent_dim)

# create the gan
gan_model = define_gan(generator, discriminator)

# summarize gan model
gan_model.summary()

# plot gan model
plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)

# train model
train(generator, discriminator, gan_model, latent_dim)
